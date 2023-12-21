# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:46:58 2020

@author: sickl

v4:
    --implemented bootstrapped standard error of the mean as the estimate for error bars.
    Error bars on shapes are supposed to be standard error of the mean because it should give
    an idea of what happens if an additional event were to be added to the averaged shape.
    The effect of each additional introduction should reduce its potency by 1/N

v5:
    --implemented a duration_shapes option. The duration_shapes option normalizes the binned velocities
    prior to averaging, while the duration option does not. This allows the average shape of avalanches
    to be observed, regardless of the absolute velocity they achieve.
    --Also ensure that all shapes begin and end at 0.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import bootstrap
import scipy
arr = np.array

#resize the (times,vels) vector pair to be a new length, length + 1, which includes both boundaries.
#Do this via interpolating missing data.
def my_resize(vels,times,length):
    tmp = np.arange(1,length-1)/(length-1)
    outtimes = np.array([0] + list(tmp) + [1])
    mytimes = (np.array(times)-np.min(times))/(np.max(times)-np.min(times))
    f = interp1d(mytimes,vels)
    outvels = f(outtimes)
    return outtimes, outvels

#find bin centers that span the whole scaling regime for a given width w.
#Works for any quantity to be averaged over (duration, size, etc)
def find_centers(smin,smax,w,nbins):
    return [(smin/(1-w))*(((1-w)*smax/((1+w)*smin))**(i/(nbins-1))) for i in list(range(nbins))]

#find the maximum width allowable for a given smin,smax and nbins. Works for
#any quantity to be averaged over (duration, size, etc)
def find_max_width(smin,smax,nbins):
    r = smax/smin
    max_width = (r**(1/nbins) - 1)/(r**(1/nbins) + 1)
    return max_width

#v3 of the shapes code. Based on j_shapes, with some optimizations built in.
def shapes(v = None, t = None, s = None, d = None, centers = None, 
           style = 'size', width  = 0.15, nbins = 5, mylimits = None, errbar_type = 'bootstrap', ci = 0.95):
    """
    Parameters
    ----------
    v : Event velocities, output from get_slips. Required.
    t : Event times, output from get_slips. Required.
    s : Event sizes, output from get_slips. Required.
    d : Event durations, output from get_slips. Required.
    centers: list, optional
        Set the bin centers manually, if you so desire. Otherwise, the program
        will determine the bin centers automatically.
    style : str, optional
        Set the type of averaging to do. Can be 'duration', 'size', or 'duration_shapes'.
        'size' = bin according to size
        'duration' = bin according to duration, then average.
        'duration_shapes' = bin according to duration, but normalize the velocity prior to averaging. Allows the shapes to be observed.
        The default is 'size'.
    width : float, optional
        The fractional half-width of each bin. Events are collected into a bin
        with a center value X_c if an event has X in the range 
        (1-width)*X_c < X < (1+width)*X_c. Default value is ~0.15.
        
        If value is set to None, tells the system to automatically choose
        the width to be maximum, which is not always the best.
        
    ***MAX WIDTH ALLOWED FOR BOUNDARIES S_min, S_max, and N bins:
        |    Let r = S_max/S_min
        |    width_max = (r^(1/N) - 1)/(r^(1/N) + 1)
        |    
        |    **BIN POSITIONS**
        |    Number each bin s_1, s_2, ... s_N. Furthermore, (1-w)*s_1 = s_min and (1+w)*s_N = smax.
        |    Then, each bin position is given by: s_1 = s_min/(1-w), s_(n-1) *(1+w)/(1-w) = s_n, 1 < n <= N.
        
        ***FINDING BIN CENTERS FOR AUTOMATIC BINNING with S_min, S_max, width w, and N bins:
            Number each bin s_1, s_2, ... s_N
            Let s_1 = s_min/(1-w) and s_N = s_max/(1+w)
            s_n = s_1*((s_N/s_1)**(n/N))
            s_n = s_min/(1-w)*((1-w)s_max/((1+w)s_min))**(n/(N-1))
        
    nbins : int, optional
        The number of bins to create. The default is 5.
    mylimits : list, optional
        A list which holds the minimum and maximum X to search for, [min(X), max(X)].
        X can either be duration or size, depending on the style chosen.
        The default is None.
        
    errbar_type: string, optional
        Either 'bootstrap' or 'std'. 'std' for standard deviation (faster but 
        less realistic for points near the edges) and 'bootstrap' for using
        a bootstrapping approach (slow, but far more accurate)
        
    ci: float, optional
        The confidence interval, between 0 and 1. Defaults to 0.95 for 95% CI

    Returns
    -------
    
    allt: 3-dimenisonal list.
        A (nbins)-long list of lists of lists which holds the individual time
        traces for the avalanches which make up each bin.
    allv: 3-dimensional list.
        A (nbins)-long list of lists of lists which holds the individual
        velocity profiles for the avalanches which make up each bin.
    avgt: list of lists of floats.
        Holds the time vectors for each of the averaged profiles.
    avgv: list of list of floats.
        Holds the velocity vector for each of the averaged profiles.
    avgstd: list of list of floats
        Holds the [lo,hi]  95% confidence intervals for each bin.
        As of v4, error bars are identified via bootstrapping.
    centers: list of floats.
        The bin centers. If binning by size, this is the center of each bin in size.
    width: float
        Fractional width width for each bin. Defined as (1-w)*x_c < x_c < (1+w)*x_c
        for center x_c and width w.

    """
    
    #perform a test to ensure the right inputs are defined
    if any(i is None for i in [v,t,s,d]):
        print('Please input the outputs from get_slips. v, t, s, and d must all be explicitly defined!')
        return -1
    
    
    #####hold quntity to average over in variable x #####
    if style == 'size':
        x = s
    elif style == 'duration':
        x = d
    elif style == 'duration_shapes':
        x = d
    else:
        #if option is not correct, return -1
        print('Please give a valid option for style! Should be size, duration, or duration_shapes.')
        return -1

    #get the limits if it is not explicitly defined.
    if mylimits is None:
        mylimits = [min(x), max(x)]
    
    #get the max width allowed for each bin
    max_width = find_max_width(mylimits[0],mylimits[1],nbins)
    
    #if width is not specified, set it to max_width.
    if width == None:
        width = max_width
    #if the width is greater than max width, return -1
    if width > max_width:
        print('Width is too large! max_width = %.3f, inputted width = %.3f' % (max_width,width))
        return -1
    
    #if centers is not defined, place bins evenly throughout the scaling regime. RECOMMENDED!
    if centers == None:
        centers = find_centers(mylimits[0],mylimits[1],width,nbins)
    

    #initialize variables to hold binned time and velocity
    allt = []
    allv = []
    avgt = []
    avgv = []
    avgstd = []
    
    #bin all events into allbinnedt and allbinnedv
    for center in centers:
        #align each bin's time with t = 0
        bt = [np.array(t[idx]) - np.min(t[idx]) for idx,val in enumerate(x) if val >= center*(1-width) and val < center*(1+width)]
        bv = [np.array(v[idx])-min(v[idx]) for idx,val in enumerate(x) if val >= center*(1-width) and val < center*(1+width)] #ensure that the shapes start and stop at zero.
        
        #the time is the longest duration in the given bin
        lens = [len(i) for i in bt]
        #print(len(bt))
        idxmax = np.argmax(lens)
        tot = len(bt)
        at = np.zeros(np.max(lens))
        av = np.zeros(np.max(lens))

        #2d arrays to hold avalanches to average into
        toav_v = np.zeros([len(bt),max(lens)]) 
        for i in range(len(bv)):
            toav_v[i,:len(bv[i])] += np.array(bv[i])
            
        #if the style is size, add all events aligned from the left (t = 0).
        if style == 'size':
            at = bt[idxmax] - min(bt[idxmax])            
        #if the style is duration or duration_shapes:
        else:
            maxlen = len(bt[idxmax])
            for i in range(len(bv)):
                #if style is duration shapes, normalize the velocities prior to averaging
                avv = bv[i]
                avt = bt[i]
                if style == 'duration_shapes':
                    avv = avv/max(avv)
                tmpt,tmpv = my_resize(avv,avt,maxlen)
                toav_v[i,:] = tmpv
            at = tmpt
            #resize the current bin so it goes from 0 < t < T for bin duration T.
            at *= center
        
        #if style is duration shapes, normalize the velocities when resizing.
            
        av = np.mean(toav_v,axis = 0)
        #astd = np.std(toav_v,axis = 0)
        
        #bootstrap errorbars
        astd = []
        if errbar_type == 'bootstrap':
            for i in range(np.max(lens)):
                cur = toav_v[:,i]
                conf = bootstrap((cur,),np.mean, confidence_level = ci).confidence_interval
                lo = av[i] - conf[0]
                hi = conf[1] - av[i]
                astd.append(arr([lo,hi]))            
        #standard deviation errorbars
        else:
            for i in range(np.max(lens)):
                cur = toav_v[:,i]
                z = scipy.stats.norm.ppf(1-ci/2)
                sigma = z*np.std(cur)
                astd.append(arr([sigma,sigma]))
            
        astd = arr(astd).transpose()
        allt.append(bt)
        allv.append(bv)
        avgt.append(at)
        avgv.append(av)
        avgstd.append(astd)

    return allt,allv,avgt,avgv,avgstd,centers,width
        
        
        
        
        
        