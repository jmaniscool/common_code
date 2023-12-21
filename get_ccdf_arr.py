#Alan Long 6/10/16
#Last edited: Alan Long 5/23/19
#Vectorized by Jordan Sickle 12/19/23

#This code takes data and returns its complementary cumulative distribution
#function (CCDF). It accepts a list or array data and returns two lists histx and
#histy, the x and y values for the ccdf.

import numpy as np

#Simpler and faster version of get_cum_dist() function.
#Currently between 50% and 200%+ speed improvement over get_cum_dist().
#Can be made faster if unique() is replaced by a jit-able function.

#Option to freely switch between two definitions of CCDF: option "dahmen" is C(x) = Pr(x >= X) (inclusive), while option "scipy" is C(x) = Pr(x > X) (exclusive)
def get_ccdf_arr(data, method = 'scipy'):
    
    data = np.array(data) #convert to array
    if len(data) == 0:
        return np.array([]),np.array([])
    
    data = data[(data > 0)*~np.isnan(data)*~np.isinf(data)]
    
    histx = np.sort(data)
    N = len(histx)
    histy = np.zeros(N)
    
    #get the unique values greater than zero, not nans and not infs
    _, counts = np.unique(histx, return_counts = True)
    
    #get cumulative counts for the unique points
    ccounts = np.cumsum(counts)
    
    indices = np.zeros(len(counts) + 1)
    indices[0] = 0
    indices[1:] = ccounts
    
    
    #Dahmen group's method. The CCDF is defined as Pr(x >= X) (includes current value)
    if method == 'dahmen':
        
        #start counting at zero
        ccounts[1:] = ccounts[0:-1]
        ccounts[0] = 0
        
        ccounts = 1 - ccounts/N
    
    k = 0
    for i in range(len(histy)):
        k += i == indices[k+1]
        histy[i] = ccounts[k]
        
    
    #if method is scipy, then use definition of CCDF as Pr(x > X) (traditional method)
    if method == 'scipy':            
        histy = 1 - histy/histy[-1]

    
    return histx,histy

