# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:24:08 2024

@author: sickl

Hold all the different distance functions (i.e. find_d_sorted, find_ad_sorted)
"""
import numba
import numpy as np
import scipy


arr = np.array

#find nearest value in an index. From StackOverflow.
#Returns the first index where array - value is closest to zero. Useful for min
@numba.njit
def find_nearest_idx(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

@numba.njit
def find_nearest_idx_discrete(data,value, first):
    """
    Find the nearest index where value is observed in sorted data. 
    In the case of a tie, returns the first index where the value 
    is obtained if first > 0, otherwise returns the last index 
    where the value is obtained.
    
    Parameters
    ----------
    data : array
        An array of sorted values.
    value : float
        The value to search for.
    first : float
        If first > 0, return the first index that is nearest to the input. Else, return the last index.

    Returns
    -------
    int
        The index where value is closest to data.

    """
    test = np.abs(data - value)
    idxs = np.where(test == np.min(test))[0]
    if first > 0:
        return idxs[0]
    return idxs[-1]

#ASSUME DATA IS SORTED AND DATA[0] = XMIN AND DATA[-1] = XMAX
#Validated against https://doi.org/10.2478/s11600-013-0154-9 on 6-12-24, though the way they define the CDF goes from 1 to 0 and thus some algebra is required to get the same result as below.

@numba.njit
def find_d_sorted(data, alpha):
    if alpha <= 1:
        return 1e12
    #x = np.sort(data) #change to just data if the input data is sorted
    x = data
    xmin = x[0]
    xmax = x[-1]

    #test if we are in a reasonable range. If not, then the CDF is approximately a delta function either at x = xmin or x = xmax. Doesn't matter for KS distance which you choose. Return 1 as an approximation.
    test_xmin = np.log10(xmin)*(-alpha+1)
    if test_xmin > 100:
        return 1
    cdf_t = (x**(1-alpha) - xmin**(1-alpha))/(xmax**(1-alpha) - xmin**(1-alpha))
    
    ecdf = np.arange(1,len(x) + 1)/len(x) #empirical CDF = (1/N, 2/N, ... 1)
    return np.amax(np.abs(ecdf-cdf_t))

#Fast implementation of numba unique, from https://github.com/numba/numba/pull/2959
@numba.njit
def numba_unique(array):
    b = np.sort(array.ravel())
    unique = list(b[:1])
    counts = [1 for _ in unique]
    for x in b[1:]:
        if x != unique[-1]:
            unique.append(x)
            counts.append(1)
        else:
            counts[-1] += 1
    return arr(unique), arr(counts)

@numba.njit
def find_d_sorted_discrete(normdata, alpha):
    """
    Find the KS distance for a discrete data, normdata, represented in units of its stepsize.

    Parameters
    ----------
    normdata : array of ints
        The data to compute KS distance for. Represented in units of stepsize, i.e. normdata = 1 for something that is one stepsize long.
        The data is also sorted, with its first element being xmin, and its last element being xmax. Inclusive.
    alpha : float
        The alpha value to test against.

    Returns
    -------
    float
        The KS distance. Always will be between 0 and 1.

    """
    if alpha <= 1:
        return 1e12
    x = normdata
    xmin = x[0]
    xmax = x[-1]
    
    vals,y = numba_unique(x)
    y = np.cumsum(y)
    ecdf = y/y[-1]
    
    #Attempted version which aligns better with ecdf
    cdf_t = np.ones(len(vals))
    inv_bot = 1/np.sum(np.arange(xmin,xmax+1)**-alpha)
    for i in range(len(vals)):
        sumrange = np.arange(xmin,vals[i] + 1)
        cdf_t[i] = np.sum(sumrange**-alpha)*inv_bot
    return np.amax(np.abs(ecdf-cdf_t))

@numba.njit
def find_ad_sorted_discrete(normdata,alpha):
    """
    Find the AD distance for a discrete data, normdata, represented in units of its stepsize.    
    Altered from the continuous case to remove integrals, based on Choulakian et al 1994
    
    Source
    [1] Choulakian et al 1994, https://www.sfu.ca/~lockhart/Research/Papers/ChoulakianLockhartStephens.pdf

    Parameters
    ----------
    normdata : array of ints
        The data to compute KS distance for. Represented in units of stepsize, i.e. normdata = 1 for something that is one stepsize long.
        The data is also sorted, with its first element being xmin, and its last element being xmax. Inclusive.
    alpha : float
        The alpha value to test against.

    Returns
    -------
    float
        The AD distance. Lower is better.

    """
    x = normdata
    xmin = x[0]
    xmax = x[-1]
    
    N = len(x)
    if alpha <= 1:
        return 1e12
    
    test_xmin = np.log10(xmin)*(-alpha+1)
    if test_xmin > 100:
        return 1e12
    
    #empirical cdf
    vals,y = numba_unique(x)
    y = np.cumsum(y)
    ecdf = y/y[-1]
    
    #theoretical cdf + theoretical probability of obtaining each value
    cdf_t = np.ones(len(vals))
    inv_bot = 1/np.sum(np.arange(xmin,xmax+1)**-alpha)
    p = np.ones(len(vals)) #p[i] holds the probability of obtaining vals[i]
    for i in range(len(vals)):
        sumrange = np.arange(xmin,vals[i] + 1)
        cdf_t[i] = np.sum(sumrange**-alpha)*inv_bot
        p[i] = inv_bot*(vals[i]**-alpha)
        
    #calculate the A2 value.
    p = p[:-1] #Only the terms up until the last one are considered, since by definition the last term has a 0/0 discontinuity
    cdf_t = cdf_t[:-1]
    ecdf = ecdf[:-1]
    
    Z = (ecdf - cdf_t)*N #Zi = sum(o_i - e_i) (sum of observed minus expected)
    A2 = (1/N)*Z*Z*p/(cdf_t*(1-cdf_t)) #Equation 3 in Choulakian et al 1994
    
    return np.sum(A2) #the last term gives nan because cdf_t = 0 at x = xmax.

#ASSUME DATA IS SORTED AND DATA[0] = XMIN AND DATA[-1] = XMAX
#use PowerLaw D which is weighted to be more sensitive towards tails. Not as performant, but could be bugged.
def find_dstar_sorted(data, alpha):
    if alpha <= 1:
        return 1e12
    #x = np.sort(data) #change to just data if the input data is sorted
    x = data
    xmin = x[0]
    xmax = x[-1]

    #test if we are in a reasonable range. If not, then the CDF is approximately a delta function either at x = xmin or x = xmax. Doesn't matter for KS distance which you choose. Return 1 as an approximation.
    test_xmin = np.log10(xmin)*(-alpha+1)
    if test_xmin > 100:
        return 1
    cdf_t = (x**(1-alpha) - xmin**(1-alpha))/(xmax**(1-alpha) - xmin**(1-alpha))
    cdf_t[0] = 1e-12 #set CDF close to 0 for first element
    cdf_t[-1] = 1-1e-12 #set CDF close to 1 for last element
    
    ecdf = np.arange(1,len(x) + 1)/len(x) #empirical CDF = (1/N, 2/N, ... 1)
    return np.amax(np.abs(ecdf-cdf_t)/np.sqrt(cdf_t*(1-cdf_t)))

#using SciPy definition of A^2, see https://github.com/scipy/scipy/blob/v1.12.0/scipy/stats/_morestats.py#L2087-L2289
#ASSUME DATA IS SORTED AND DATA[0] = XMIN AND DATA[-1] = XMAX
@numba.njit
def find_ad_sorted(data,alpha):
    ln = np.log
    x = data
    xmin = x[0]
    xmax = x[-1]
    
    N = len(x)
    if alpha <= 1:
        return 1e12
    
    test_xmin = np.log10(xmin)*(-alpha+1)
    if test_xmin > 100:
        return 1e12
    
    idx = np.arange(1, N + 1)
    
    cdf = (x**(1-alpha) - xmin**(1-alpha))/(xmax**(1-alpha) - xmin**(1-alpha))
    cdf[0] = 1e-12 #set CDF to be close to 0 for x close to xmin
    cdf[-1] = 1-1e-12 #set cdf to be close to 1 for x close to xmax
    ccdf = 1 - cdf #CCDF for PL
    
    A2 = -N - np.sum((2*idx - 1.0) / N * (ln(cdf) + ln(ccdf)[::-1]), axis=0)
    
    return A2

#find the Kramer-Von Mises U value comparing between the empirical CDF and the power law cdf.
#Works about as well as KS distance.
def find_u_sorted(data,alpha):
    #ASSUME DATA IS SORTED AND DATA[0] = XMIN AND DATA[-1] = XMAX
    ln = np.log
    x = data
    xmin = x[0]
    xmax = x[-1]
    if alpha <= 1:
        return 1e12
    
    test_xmin = np.log10(xmin)*(-alpha+1)
    if test_xmin > 100:
        return 1e12
    
    def cdf(x):
        return (x**(1-alpha) - xmin**(1-alpha))/(xmax**(1-alpha) - xmin**(1-alpha))
    
    return scipy.stats.cramervonmises(data,cdf).statistic

#ASSUME DATA IS SORTED AND DATA[0] = XMIN AND DATA[-1] = XMAX
#Use Kuiper statistic as the variable to minimize; may give sensitivity to tails and median.
#See https://en.wikipedia.org/wiki/Kuiper%27s_test

#Limited testing suggests that this is the best one! Requires at least ~1000 or so observations in the scaling regime to get good estimates of xmin or xmax.
def find_v_sorted(data, alpha):
    if alpha <= 1:
        return 1e12
    #x = np.sort(data) #change to just data if the input data is sorted
    x = data
    xmin = x[0]
    xmax = x[-1]
    
    N = len(x)

    #test if we are in a reasonable range. If not, then the CDF is approximately a delta function either at x = xmin or x = xmax. Doesn't matter for KS distance which you choose. Return 1 as an approximation.
    test_xmin = np.log10(xmin)*(-alpha+1)
    if test_xmin > 100:
        return 1

    cdf_t = (x**(1-alpha) - xmin**(1-alpha))/(xmax**(1-alpha) - xmin**(1-alpha))
    
    #D_plus = max(ecdf - cdf_t)
    #D_minus = max(cdf_t - ecdf_reversed) #range on reverse eCDF is ((N-1)/N, (N-2)/N, ... 0)
    
    #from AstroPy
    D = np.amax(cdf_t- np.arange(N) / float(N)) + np.amax(
        (np.arange(N) + 1) / float(N) - cdf_t)
    
    return D
    
#TESTING HIGHER ORDERS OF KUIPER-LIKE STATISTICS.
def find_v2_sorted(data, alpha):
    if alpha <= 1:
        return 1e12
    #x = np.sort(data) #change to just data if the input data is sorted
    x = data
    xmin = x[0]
    xmax = x[-1]
    
    N = len(x)

    #test if we are in a reasonable range. If not, then the CDF is approximately a delta function either at x = xmin or x = xmax. Doesn't matter for KS distance which you choose. Return 1 as an approximation.
    test_xmin = np.log10(xmin)*(-alpha+1)
    if test_xmin > 100:
        return 1
    cdf_t = (x**(1-alpha) - xmin**(1-alpha))/(xmax**(1-alpha) - xmin**(1-alpha))
    
    ecdf = np.arange(1,len(x) + 1)/len(x) #empirical CDF = (1/N, 2/N, ... 1)
    ecdf_reversed = np.arange(0,len(x)) #empirical CDF = ((N-1)/N, (N-2)/N, ... 0)
    
    #D_plus = max(ecdf - cdf_t)
    #D_minus = max(cdf_t - ecdf_reversed) #range on reverse eCDF is ((N-1)/N, (N-2)/N, ... 0)
    
    dplus_srt = np.sort(cdf_t- np.arange(N) / float(N))
    dminus_srt = np.sort((np.arange(N) + 1) / float(N) - cdf_t)
    
    #from AstroPy
    D = dplus_srt[-1] + dminus_srt[-1] + dplus_srt[-2] + dminus_srt[-2] #something that feels kind of "quadratic"
    
    return D
