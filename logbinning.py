# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:49:21 2019
Log bin the data for clarity

Updated July 27 2023 for speed

@author: sickl
"""
import numpy as np
import numba
import scipy #calculate the amount to multiply error by using normal distribution ppf

arr = np.array

#difference between this version and the older one is in where the first index is placed for the bin edge.
#logbinning puts the first bin edge at unsorted_x[2], while logbinning2 puts the first bin edge at the smallest value larger than 0 (more correct).

#V2: Update to have either standard error of the mean (SEM) or standard error (SE) based on the application.
#Generally, prefer to use standard error of the mean because the error bar should describe how much the average value would change when binning data together on a logarithmic bin.
#Furthermore, report 95% CI using Z-scores.

#use matrix calculations, parallelization, and numba njit compiling to make this as fast as possible in Python. About 10-40x faster than logbinning, and a bit more accurate.
@numba.njit(parallel = True)
def logbinning_core(unsorted_x,unsorted_y,numBins, error_type = 'SEM'):
    
    #define outputs
    centers = np.zeros(numBins)
    errs = np.zeros(numBins)
    out = np.zeros(numBins)
    
    unsorted_y = unsorted_y[unsorted_x > 0] #get only positive values
    unsorted_x = unsorted_x[unsorted_x > 0]
    
    
    
    idxs = np.argsort(unsorted_x)
    
    #organize by first index
    x = unsorted_x[idxs]
    y = unsorted_y[idxs]
    
    logmax = np.log10(x[-1])
    logmin = np.log10(x[0])
    
    #get edges
    edges = np.logspace(logmin,logmax,numBins + 1)
    #get edge indices
    edgeidxs = np.zeros(numBins + 1)
    for i in range(numBins + 1):
        tmp = np.abs(x - edges[i])
        edgeidxs[i] = tmp.argmin() #find minimimum from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        
    
    #get centers
    dx = (logmax-logmin)/numBins
    centers = np.logspace(logmin + dx, logmax - dx, numBins)
    
    #get means
    for i in range(numBins):
        st = int(edgeidxs[i])
        en = int(edgeidxs[i + 1])
        
        #add 1 to take into account when start and end are same index
        en = en + int(st == en)
        vals = y[st:en]
        out[i] = np.mean(vals)
        if error_type == 'SEM':
            errs[i] = np.std(vals)/np.sqrt(en-st) #SEM = std(X)/sqrt(N). N = en-st.
        else:
            errs[i] = np.std(vals) #standard error = std(X)
        
    return centers,out,errs

def logbinning(unsorted_x,unsorted_y, numBins, error_type = 'SEM', ci = 0.68):
    centers, out, errs = logbinning_core(unsorted_x,unsorted_y, numBins, error_type = error_type)
    
    z = np.sqrt(2)*scipy.stats.norm.ppf((1 + ci)/2)
    return centers, out, errs*z
    

"""
#old version of logbinning. A bit slower.
def logbinning(unsorted_x,unsorted_y,numBins):
    
    #create data in a tuple, (x,y), sort according to the first value, then reassign them to x,y
    tmp = []
    for i in range(len(unsorted_x)):
        tmp.append((unsorted_x[i],unsorted_y[i]))
    tmp = sorted(tmp,key=lambda x: x[0])
    #remove negative values
    tmp = [(i[0],i[1]) for i in tmp if i[0] > 0]
    
    x = []
    y = []
    for i in range(len(tmp)):
        x.append(tmp[i][0])
        y.append(tmp[i][1])
    
    logMax = np.log10(max(x))
    logMin = np.log10(x[1]) #hotfix to make sure the t=0 error is avoided
    binEdges = 10**np.linspace(logMin,logMax,numBins + 1)
    idxBinEdges = find_index(binEdges,x)
    
    #convert to values
    binCenters = []
    out= []
    err = []
    
    #define the bin centers
    for i in range(1,len(binEdges)):
        binCenters.append(10**((np.log10(binEdges[i]) + np.log10(binEdges[i-1]))/2))
    
    #Take the mean of the data that resides in each bin
    for i in range(1,numBins + 1):
        idx_start = int(idxBinEdges[i-1])
        idx_end = int(idxBinEdges[i]) #add 1 to take into account when start and end are same index
        if idx_start == idx_end:
            idx_end = idx_end + 1
        out.append(np.mean(y[idx_start:idx_end]))
        #doing the error as dy/y
        #err.append(1.96*np.std(y[idx_start:idx_end])/np.mean(y[idx_start:idx_end]))
        #doing the standard error
        err.append(np.std(y[idx_start:idx_end])/np.sqrt(idx_end-idx_start))
    
    #binCenters = np.log10(binCenters).tolist()
    #out = np.log10(out).tolist()
    
    return binCenters,out,err

#find the index of the sorted data
def find_index(nums,data):
    out = []
    for curNum in nums:
        i = 0
        while data[i] < curNum:
            if i < len(data)-1:
                i+=1
            else:
                break
        out.append(i)
    return out
"""
2+2