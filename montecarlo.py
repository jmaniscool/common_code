# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:23:05 2024
#include all the monte-carlo estimators (i.e. find_pl_montecarlo), finding p value thresholds, etc.
"""
import numba
import numpy as np
from .distances import find_d_sorted, find_ad_sorted, find_nearest_idx
from .brent import brent_findmin
from .likelihoods import pl_gen

#used to estimate the pq value from D in the monte carlo xmin/xmax. 10 terms is plenty to get extremely high accuracy.
@numba.njit
def expfun(x,numterms = 10):
    val = 0
    for i in range(1,numterms+1):
        val = val + (-1)**(i-1)*np.exp(-2* i**2 * x**2)
        
    return 2*val

#find the true p value (as compared to pq). See Deluca & Corrall 2013 (https://doi.org/10.2478/s11600-013-0154-9).
#If true p > 0.15 (or maybe p > 0.2), then the power law fit is good.
@numba.njit
def find_true_p(x,xmin,xmax,runs = 150, dfun = find_d_sorted):
    
    tot = 0
    p = 1
    sigp = 0
    xmin_idx = find_nearest_idx(x,xmin)
    xmax_idx = find_nearest_idx(x,xmax)
    x = x[xmin_idx:xmax_idx + 1]
    alpha = brent_findmin(x)
    de = dfun(x,alpha)
    for i in range(runs):
        synth = np.sort(pl_gen(len(x),xmin,xmax,alpha))
        asynth = brent_findmin(synth)
        ds = dfun(synth,asynth)
        tot = tot + (ds >= de) #if ds > de, increment tot
        
    p = tot/runs
    sigp = np.sqrt(p*(1-p)/runs) #1 sigma (68% CI)
    
    return p,sigp
#core of the find_p part of the montecarlo code. Broken into its own jitted function to improve overhead in communicating between C and Python.
#very minimal speed increase (<10%) compared to python implementation.
@numba.njit
def find_p_core(data,possible_xmins,possible_xmaxs, pruns, dfun):
    possible_ps = np.zeros(len(possible_xmins))
    for i in range(len(possible_ps)):
        xmin = possible_xmins[i]
        xmax = possible_xmaxs[i]
        xmin_idx = find_nearest_idx(data,xmin)
        xmax_idx = find_nearest_idx(data,xmax)
        trimmed = data[xmin_idx:xmax_idx + 1]
        possible_ps[i] = find_true_p(trimmed,xmin,xmax, runs = pruns, dfun = dfun)[0]
    
    return possible_ps
        
    
#use a monte carlo approach of finding xmin, xmax, and alpha using KS statistic.
def find_pl_montecarlo(data, runs = 2000, pqcrit = 0.35, pcrit = 0.2, pruns = 100, dist_type = 'KS', calc_p = False):
    """
    Use a Monte Carlo approach to find xmin,xmax, and alpha using KS statistics.
    Calculate KS distance for [runs] samples of xmin/xmax from [data]. Return the run where xmax/xmin is largest and pq > pcrit.
    Parameters:
        data (np.array or list): The data to find the power law from.
        runs (int): The number of random choices of xmin/xmax to calculate pq from.
        pqcrit (float): the critical value for pq; a run is labeled as possible if pq > pcrit
        pcrit (float): the critical value for true p.
        dist_type (string): either KS for Kolmogorov-Smirnov or AD for Anderson-Darling tests.
        
    """
    
    data = np.sort(data) 
    dfun = find_d_sorted
    defaults = [1,0,0,0,0]
    if calc_p == False:
        pcrit = pqcrit
        
    if dist_type == 'KS':
        dfun = find_d_sorted
    elif dist_type == 'AD':
        dfun = find_ad_sorted
    else:
        print('Error. Please input a valid distance type (either KS or AD)')
        return defaults

    #As of 6-12-24, use an updated method that searches for any with pq > 0.35 (gives p approx 0.15 or so) then selects the run that maximizes xmax/xmin
    #NOTE: due to a bug in numba (conditional statements don't work in for loops), we cannot compile this core function to njit.
    attempted_xmins = np.ones(runs)
    attempted_xmaxs = np.ones(runs)
    attempted_alphas = np.ones(runs)
    attempted_ns = np.ones(runs)
    attempted_pqs = np.ones(runs)
    attempted_ds = 1e12*np.ones(runs)
    log_xmin = np.log(data[0])
    log_xmax = np.log(data[-1])
    log_range = log_xmax-log_xmin
    trial_xmin = 0
    trial_xmax = 0
    trial_xmin_idx = 0
    trial_xmax_idx = 0
    for i in range(runs):
        trial_xmin = np.exp(log_xmin + log_range*np.random.rand())
        trial_xmax = np.exp(log_xmin + log_range*np.random.rand())
        trial_xmin_idx = find_nearest_idx(data,trial_xmin)
        trial_xmax_idx = find_nearest_idx(data,trial_xmax)
        trial_xmin = data[trial_xmin_idx]
        trial_xmax = data[trial_xmax_idx]
        
        #ensure there are at least 10 datapoints and that xmax > 2*xmin
        #the comparison in the while loop is what ruins being able to njit this function. It cannot be replaced by anything that has a conditional in it.
        while (trial_xmax_idx - trial_xmin_idx < 10) or (trial_xmax < 2*trial_xmin):
            trial_xmin = np.exp(log_xmin + log_range*np.random.rand())
            trial_xmax = np.exp(log_xmin + log_range*np.random.rand())
            trial_xmin_idx = find_nearest_idx(data,trial_xmin)
            trial_xmax_idx = find_nearest_idx(data,trial_xmax)
            trial_xmin = data[trial_xmin_idx]
            trial_xmax = data[trial_xmax_idx]
        trimmed = data[trial_xmin_idx:trial_xmax_idx+1]
        alpha_hat = brent_findmin(trimmed)
        tmpd = 1
        if alpha_hat > 1:
            attempted_ds[i] = dfun(trimmed,alpha_hat)
            tmpd = find_d_sorted(trimmed,alpha_hat)
        n = len(trimmed)
        
        
        #Function is Equation 29 in Deluca & Corrall 2013 (https://doi.org/10.2478/s11600-013-0154-9)
        #and is given on the wikipedia article on the KS test. According to Deluca & Corrall 2013, this only weakly correlates with the true p value.
        #this is obtained by noting that expfun(z) is distributed according to the expfun distribution if one sets z = np.sqrt(z)*d + (correction factors).
        #attempted_pqs[i] = expfun(tmpd*np.sqrt(n) + 0.12*tmpd + 0.11*tmpd/np.sqrt(n))
        
        #By first converting d --> d*np.sqrt(n), then we use the updated term from DOI: 10.4236/am.2020.113018 by Jan Vrbik 2020 (much more accurate)
        tmpd = tmpd*np.sqrt(n) 
        attempted_pqs[i] = expfun(tmpd + 0.17/np.sqrt(n) + (tmpd - 1)/(4*n))
        attempted_ns[i] = n
        attempted_alphas[i] = alpha_hat
        attempted_xmins[i] = trial_xmin
        attempted_xmaxs[i] = trial_xmax
    
    #Depreciated as of 6-12-24. The best index is found as the one that maximizes xmax/xmin while being above pq > pcrit
    #minidx = np.argmin(attempted_ds)
    
    #find the possible indices. For each possible index, calculate the true p value using simulation.
    idxs = np.where(attempted_pqs > pqcrit)[0]
    possible_xmins = attempted_xmins[idxs]
    possible_xmaxs = attempted_xmaxs[idxs]
    possible_alphas = attempted_alphas[idxs]
    possible_pqs = attempted_pqs[idxs]
    #print(len(possible_pqs))
    
        
    possible_ps = np.zeros(len(possible_pqs))
    if calc_p == True:
        possible_ps = find_p_core(data,possible_xmins,possible_xmaxs, pruns, dfun)
    else:
        possible_ps = possible_pqs
    
    #only examine runs where the p-value is greater than the pcrit (default 0.2)
    idxs2 = np.where(possible_ps > pcrit)[0]
    possible_xmins2 = possible_xmins[idxs2]
    possible_xmaxs2 = possible_xmaxs[idxs2]
    possible_alphas2 = possible_alphas[idxs2]
    possible_pqs2 = possible_pqs[idxs2]
    possible_ps2 = possible_ps[idxs2]
    #print(len(possible_ps2))
    
    minidx = np.argmax(possible_xmaxs2/possible_xmins2)
    
    xmin = possible_xmins2[minidx]
    xmax = possible_xmaxs2[minidx]
    alpha = possible_alphas2[minidx]
    
    #the "false" p-value from equation 29 of (https://doi.org/10.2478/s11600-013-0154-9).
    pq = possible_pqs2[minidx]
    p = possible_ps2[minidx]
    

    return alpha,xmin,xmax,pq,p,len(possible_pqs),len(possible_ps2)