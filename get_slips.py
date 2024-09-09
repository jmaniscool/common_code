#Alan Long 6/14/16
#Last edited: Jordan Sickle 10/21/23

#Past versions by Tyler Salners, Alan Long, Jim Antonaglia, with version as of 10/21/23 updated by Jordan Sickle.

#Code explanations.
"""
This code will search for avalanches as points in the 'velocity' curve which exceed a threshold.
The size is the amount of displacement (decrease if drops == True) that occurs while the velocity is above a threshold.

If displacement is given, the code will do a numerical derivative using np.diff to find the velocity.
If velocity is given, the code will do a numerical integral using np.cumsum to find the displacement.
If time vector is not given, the code will assume a time vector composed of index numbers. (i.e. the avalanche velocity is in terms of displacement per timestep)

One of either displacement/velocity is required to be given, or both.

Inputs
--------

displacement: None or vector of floats
    The displacement curve (i.e. curve before you do a derivative) to find avalanches in.
    Useful if no filtering is done on data (i.e. in simulation, or in special occasions for data)
    Must include at least one of displacement or velocity.

velocity: None or vector of floats.
    The velocity curve (i.e. the curve after a derivative). Avalanches are identified as
    parts of this curve which exceed 'threshold' standard deviations above the average.
    Recommend giving this after Weiner filtering your data, if available.
    
time: None or vector of floats
    The time vector. If not given, assumes time = vector of indices.
    
drops: Boolean, default True.
    True if looking for drops in the velocity curve, or False if looking for increases as avalanches.

threshtype: string, default 'median'
    --Setting 'mean' is the traditional method. The threshold is compared to the 
    mean velocity, (displacement[end]-displacement[start])/(time[end]-time[start]).
    Threshold is calculated using the standard deviation with this setting.
    This method has some major issues in non-simulation environments, as the
    standard deviation is very sensitive to large excursions from the noise floor
    (i.e. if there are many avalanches).
    
    --Setting 'median' uses the median velocity instead of the mean velocity to
    quantify the average velocity. Works best in signals with many excursions
    (i.e. many avalanches) and is not sensitive to outliers.
    Threshold is calculated using median absolute deviation (MAD) with this
    option instead of standard deviation because it more accurately describes
    the dispersion of the noise fluctuations while ignoring the avalanches.
    
    --Setting 'sliding_median' uses a sliding median of width window_length to obtain
    a sliding median estimate of the background velocity. Useful when the background
    rate of the process is not constant over the course of the experiment.
    Threshold is calculated using median absolute deviation with this option 
    and follows the change in average velocity while accurately describing the
    dispersion of the noise.

shapes: int, optional. Default 1.
    Tells the code if the avalanche 'shape' (i.e. velocity versus time) curves
    should be returned for all avalanches. E.g. v[0] is the velocity obtained for the first avalanche, and t[0] is the corresponding time vector.

window_size: int, optional. Default 101.
    The window size, in datapoints, to be used when calculating the sliding median. The window length
    should be set to be much longer than the length of an avalanche in your data.
    Jordan found setting window_size = 3% the length of the data (in one case) was
    good, but this value can be anywhere from just a few hundred datapoints (very short avalanches, many datapoints)
    to up to 10-20% of the total length of the signal (when the data are shorter but contain several very long avalanches).
    This is worth playing with!

threshold: float, optional. Default 0.
    number of standard deviations above the average velocity a fluctuation
    must be before it is considered an event. Recommend 0 for most applications
    (threshold follows the average velocity). -1 forces threshold = 0 on velocity curve,
    useful for simulations.
    
mindrop: float, optional. Default 0.
    Minimum size required to be counted as an event. Recommend 0 when starting
    analysis, but should be set higher when i.e. data culling to find the true
    value of tau.
    
Returns
---------

velocity: list of lists
    A list of avalanche velocity curves. E.g. v[8] will be the velocity curve for the 9th avalanche.
    Velocity curve will always begin and end with 0s because the velocity had to cross 0 both times in its run.

times: list of lists
    A list of avalanche time curves. E.g. t[8] will be the corresponding times at which v[8] velocity curve occured.
    The time curve will have the 0s in the velocity curve occur at t_start - ts/2 and t_end + ts/2 as a 0-order estimate
    of when the curve intersected with zero.
    
sizes: list of floats
    List of avalanche sizes. The size is defined as the amount the displacement changes while the velocity is strictly above the threshold.
    Each size is corrected by the background rate. That is, (background rate)*(duration) is removed from the event size.

durations: list of floats
    list of avalanche durations. The duration is the amount of time the velocity is strictly above the threshold.
    For example, an avalanche with velocity profile [0,1,2,3,2,1,0] has a duration of 5 timesteps.

st: list of ints
    list of avalanche start indices. E.g. st[8] is the index on the displacement where the 9th avalanche occurred.
    The start index is the first index that the velocity is above the threshold.

en: list of ints
    list of avalanche end indices. E.g. en[8] will be the index on the displacement where the 9th avalanche ends.
    The end index is the first index after the velocity is below the threshold (accounting for Python index counting)
    

"""

#Jordan edits
#v2: edited to get rid of ragged array output.
#v3: edited to deal with very low time resolution appropriately
#v4: edited to appropriately consider start/end of avalanches when considering size.
        #Also, use sliding median to get time varying threshold, and median absolute difference to get the standard deviation of noise


import numpy as np
from scipy import integrate
import scipy
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import stats
import numba

arr = np.array #shorthand for "array"

#like standard deviation but for median
mad = scipy.stats.median_abs_deviation
sliding_median = scipy.ndimage.median_filter

#wrapper function for the vectorized and fixed version of get_slips above.
def get_slips_wrap(displacement = None, velocity = None, time = None, drops = True, threshold = 0, mindrop = 0, shapes = 1, threshtype = 'median', window_size = 101):
    
    
    #if time is not given, use time = np.arange(len(sig))
    is_integrated = 0
    
    window_size = window_size + (1 - window_size % 2) #make window_size the nearest odd number for compatability with sliding_median
    
    if displacement is None and velocity is None:
        print("Error! Give at least one of velocity or displacement.")
        return -1
    
    if time is None and displacement is None:
        time = np.arange(0,len(velocity))
        
    if time is None and velocity is None:
        time = np.arange(0,len(displacement))
    
    #velocity input
    #st and en indices are the start and end in velocity-land
    #the time vector is shifted forward by 1 index so the displacement and time are same length
    if displacement is None:
        displacement = scipy.integrate.cumulative_trapezoid(velocity,time)
        displacement = np.append([displacement[0], displacement[0]], displacement)
        dt = np.median(np.diff(time))
        time = np.append(time[0],time + dt) #make time and displacement the same length
        is_integrated = 1 
        
        #displacement start index is index_velocity_begins + 1
        #displacement end index is index_velocity_ends + 1
        
    if threshtype == 'mean':
        threshtype = 1
    elif threshtype == 'median':
        threshtype = 2
    elif threshtype == 'sliding median' or 'sliding_median':
        threshtype = 3
    else:
        print("Error! The threshold type must be either mean, median, sliding median, or sliding_median")
        return
        

    #displacement input
    #st and en indices are start and end in displacement-land
    if velocity is None:
        velocity = np.diff(displacement)/np.diff(time)    
        
        #displacement start index is index_velocity_begins + 1
        #displacement end index is index_displacement_ends
        

    #if looking at drops, invert the signal.
    displacement = ((-1)**drops)*arr(displacement)
    velocity = ((-1)**drops)*arr(velocity)
    
    #print('velocity length = %d' % (len(velocity)))
    #print('displacement length = %d' % (len(displacement)))
    #print('time length = %d' % (len(time)))
    
    #minlen = min([len(displacement), len(velocity)])
    #time = np.array(time[:minlen])
    #displacement = np.array(displacement[:minlen])
    #velocity = np.array(velocity[:minlen])
    
        
    outs = get_slips_core(displacement, velocity, time, threshold, mindrop, shapes, is_integrated, threshtype, window_size)
    return outs


#Tyler's get slips, but takes in displacement and velocity. Vectorized for speed and various off-by-one errors are fixed.
#time and smoothed should be length N, deriv should be length N-1
def get_slips_core(smoothed, deriv, time, threshhold, mindrop, shapes, is_integrated, threshtype, window_size):
    #The factor of -1 is so that we deal with positive numbers
    #for drop rates, it's not strictly neccessary but makes things nicer.
    
    smoothed = np.array(smoothed)
    deriv = np.array(deriv)

    #We now take a numeric derivative and get an average
    #diff_avg=(smoothed[-1]-smoothed[0])/(time[-1]-time[0])
    #
    if threshtype == 1:
        diff_avg = np.mean(deriv)
        diff_avg = np.ones(len(deriv))*diff_avg
    elif threshtype == 2:     #do pure median
        diff_avg = np.median(deriv) #works better at getting the true size
        diff_avg = np.ones(len(deriv))*diff_avg
    else:
        diff_avg = sliding_median(deriv, window_size, mode = 'nearest')
        diff_avg[:window_size//2] = diff_avg[window_size//2]
        diff_avg[-window_size//2:] = diff_avg[-window_size//2]
        
        
    #We now set the minimum slip rate for it to be considered an avalanche

    if threshhold==-1:
        #min_diff = 0.;
        min_diff = np.zeros(len(deriv)) #make sure min_diff is a vector
    else:
        #print('hi',np.std(deriv),diff_avg)
        
        #old code for mean
        #min_diff = diff_avg + np.std(deriv)*threshhold;
        
        if threshtype == 1:
            min_diff = diff_avg + np.std(deriv)*threshhold
            
        else:
            #works for both cases of sliding median and overall median
            #new code using median average difference (see https://en.wikipedia.org/wiki/Median_absolute_deviation)
            #estimates the standard deviation of the noisy part of the data by assuming noise is normally distributed
            #and using relationship between MAD and standard deviation for normal data
            min_diff = diff_avg + 1.4826*threshhold*mad(deriv - diff_avg)


    #Now we see if a slip is occurring, i.e. if the derivative is above the
    #minimum value. We then take the diff of the bools to get the starts and ends. +1= begin, -1=end.
    
    #shift by one index forward
    slips = np.append(0,np.diff(1*(deriv > min_diff)))
    velocity_index_begins = np.where(slips == 1)[0] #velocity start index (first index above 0)
    velocity_index_ends = np.where(slips == -1)[0]  #velocity end index (last index above 0)

    #We must consider the case where we start or end on an avalanche, this
    #checks if this is case and if so makes the start or end of the data
    #a start or end of an avalanche
    if velocity_index_begins.size == 0:
        velocity_index_begins = np.array([0])
    if velocity_index_ends.size == 0:
        velocity_index_ends = np.array([len(time)-1])
    if velocity_index_begins[-1] >= velocity_index_ends[-1]:
        velocity_index_ends = np.append(velocity_index_ends, len(time) - 1)
    if velocity_index_begins[0] >= velocity_index_ends[0]:
        velocity_index_begins = np.insert(velocity_index_begins, 0, 0)
        
    #correcting for if displacement is integrated or not.

    #in all cases, the displacement index is one more than velocity index begins
    displacement_index_begins = velocity_index_begins
    
    displacement_index_ends = velocity_index_ends #+ is_integrated
    
    #reported start and end index different depending on if displacement or velocity is given
    index_begins = is_integrated*velocity_index_begins + (1-is_integrated)*displacement_index_begins
    index_ends = is_integrated*velocity_index_ends + (1-is_integrated)*displacement_index_ends
    
        
        

    #Now we see if the drops were large enough to be considered an avalanche
    
    #avalanche duration calculated from signal input
    #mindrop_correction = diff_avg*(time[index_ends - 1]-time[index_begins])*int(threshhold!=-1) #correction term which removes background rate
    
    #mindrop_correction = diff_avg*(time[index_ends - is_integrated]-time[index_begins])*int(threshhold!=-1)
    
    #first order approximation assuming the background velocity is constant throughout the avalanche
    mindrop_correction = diff_avg[index_begins]*(time[index_ends - is_integrated]-time[index_begins])*int(threshhold!=-1)
    
    index_av_begins = index_begins[mindrop < (smoothed[displacement_index_ends] - smoothed[displacement_index_begins] - mindrop_correction)]
    index_av_ends = index_ends[mindrop < (smoothed[displacement_index_ends] - smoothed[displacement_index_begins] - mindrop_correction)]

    #Finally we use these indices to get the durations and sizes of the events, accounting for diff()
    slip_durations= time[index_av_ends]-time[index_av_begins]
    dt = np.median(np.diff(time))
    
    #mindrop correction term. Term goes to 0 if avalanche is a single time step, leading to LeBlanc et al 2016 definition of size.
    duration_calculation = time[index_av_ends - is_integrated]-time[index_av_begins]
    #duration_calculation
    
    #increment size calculation by 1 if the displacement was integrated to account for cumtrapz()
    is_step = 1*((slip_durations <= dt)*(is_integrated)*(index_av_ends < len(smoothed)-1))
    
    
    #sizes are more accurately reported by only correcting for background rate, not the rate + threshold
    slip_sizes = smoothed[index_av_ends + is_step]- smoothed[index_av_begins] - diff_avg[index_av_begins]*duration_calculation*int(threshhold != -1)
    #slip_sizes = smoothed[index_av_ends + is_step]- smoothed[index_av_begins] - min_diff[index_av_begins]*duration_calculation*int(threshhold != -1)
    time_begins = time[index_av_begins]
    time_ends = time[index_av_ends]
    time2=0.5*(time[0:len(time)-1]+time[1:len(time)])
    tsamp = np.median(np.diff(time2)) #sampling time
    if shapes==1:
        velocity = []
        times = []
        for k in range(len(index_av_begins)):
            #mask= np.arange(index_av_begins[k] - (index_av_begins[k] == len(deriv)),index_av_ends[k] + (slip_durations[k] < 1))
            
            st = index_av_begins[k]
            en = index_av_ends[k]
            
            mask= np.arange(st,en)
            
            if st == en:
                mask = st
            


            #curv = deriv[mask] - min_diff #remove the min_diff
            #curt = time2[mask]
            #first order approximation: assume the shape begins and ends at min_diff halfway between the start index and the preceeding index. 
            curv = np.zeros(en-st + 2)
            curt = np.zeros(en-st + 2)
            curv[1:-1] = deriv[mask] - min_diff[st]
            #curv = np.concatenate(([0],curv,[0]))
            curt[1:-1] = time2[mask]
            curt[0] = curt[1]-tsamp/2
            curt[-1] = curt[-2] + tsamp/2
            #stt = curt[0] - tsamp/2
            #ent = curt[-1] + tsamp/2 #assume 1 index forward
            #curt = np.concatenate(([stt], curt, [ent]))
            velocity.append(list(curv))
            times.append(list(curt))
        return [list(velocity), list(times), list(slip_sizes),list(slip_durations),list(index_av_begins), list(index_av_ends)]
    else:
        return [list(slip_sizes),list(slip_durations),list(index_begins), list(index_ends)]
    
    
#use the trapezoidal rule + chopping off all parts of the signal less than the threshold (i.e. velocity < threshold*std(data) or whatever) to get a more accurate view of the size of an event.
def get_slips_vel(time, velocity, drops = True, threshold = 0, mindrop = 0, threshtype = 'median', window_size = 101):

    trapz = scipy.integrate.trapezoid
    std = lambda x : 1.4826*mad(x)    
    if threshtype == 'median':
        avg = np.median
    if threshtype == 'mean':
        avg = np.mean
        std = np.std
    cutoff_velocity = (avg(velocity) + std(velocity)*threshold*(int(threshold != -1)))*np.ones(len(velocity))
    if threshtype == 'sliding_median':
        cutoff_velocity = sliding_median(velocity, window_size, mode = 'nearest')
        cutoff_velocity[:window_size//2] = cutoff_velocity[window_size//2]
        cutoff_velocity[-window_size//2:] = cutoff_velocity[-window_size//2]
        cutoff_velocity = cutoff_velocity + std(velocity)*threshold*(int*threshold != -1)
        
    
    #treat the velocity by removing the trend such that its centered around zero
    deriv = velocity - cutoff_velocity
    if drops == True:
        deriv = -velocity

    #search for rises in the deriv curve
    
    #set all parts of the curve with velocity less than zero to be equal to zero
    deriv[deriv < 0] = 0
    
    #get the slips
    slips = np.append(0,np.diff(1*(deriv > 0)))
    index_begins = np.where(slips == 1)[0] #velocity start index (first index above 0)
    index_ends = np.where(slips == -1)[0]  #velocity end index (last index above 0)
    
    if index_begins.size == 0:
        index_begins = np.array([0])
    if index_ends.size == 0:
        index_ends = np.array([len(time)-1])
    if index_begins[-1] >= index_ends[-1]:
        index_ends = np.append(index_ends, len(time) - 1)
    if index_begins[0] >= index_ends[0]:
        index_begins = np.insert(index_begins, 0, 0)
    
    #get the possible sizes
    possible_sizes = np.zeros(len(index_begins))
    possible_durations = np.zeros(len(index_begins))
    for i in range(len(index_begins)):
        st = index_begins[i]
        en = index_ends[i]
        trapz_st = max([st-1,0])
        trapz_en = min(en + 1, len(deriv))
        possible_sizes[i] = trapz(deriv[trapz_st:trapz_en], time[trapz_st:trapz_en])
        possible_durations[i] = time[en]-time[st]
        
    idxs = np.where(possible_sizes > mindrop)[0]
    sizes = possible_sizes[idxs]
    durations = possible_durations[idxs]
    index_av_begins = index_begins[idxs]
    index_av_ends = index_ends[idxs]

    time2=0.5*(time[0:len(time)-1]+time[1:len(time)])
    tsamp = np.median(np.diff(time2)) #sampling time
    time2 = np.append(time2,time2[-1] + tsamp)
    velocity = []
    times = []
    for k in range(len(index_av_begins)):
        #mask= np.arange(index_av_begins[k] - (index_av_begins[k] == len(deriv)),index_av_ends[k] + (slip_durations[k] < 1))
        
        st = index_av_begins[k]
        en = index_av_ends[k]
        
        mask= np.arange(st,en)
        
        if st == en:
            mask = st
        


        #curv = deriv[mask] - min_diff #remove the min_diff
        #curt = time2[mask]
        #first order approximation: assume the shape begins and ends at min_diff halfway between the start index and the preceeding index. 
        curv = np.zeros(en-st + 2)
        curt = np.zeros(en-st + 2)
        curv[1:-1] = deriv[mask]
        #curv = np.concatenate(([0],curv,[0])
        curt[1:-1] = time2[mask]
        curt[0] = curt[1]-tsamp/2
        curt[-1] = curt[-2] + tsamp/2
        velocity.append(list(curv))
        times.append(list(curt))
    
    return [list(velocity), list(times), list(sizes),list(durations),list(index_av_begins), list(index_av_ends)]