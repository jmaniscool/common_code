# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:00:30 2022

@author: sickl
**WHEN USING THIS PACKAGE, YOU MUST CITE THE FOLLOWING SOURCES:
    [1] J. Alstott, E. Bullmore, D. Plenz, PLoS One 9 (2014) 85777.
    [2] A. Clauset, C.R. Shalizi, M.E.J. Newman, Http://Dx.Doi.Org/10.1137/070710111 51 (2009) 661–703.

Performs the analysis of Clauset et al 2009, with some methodology, functions, and inspiration taken from
the python powerlaw() package by Alstott et al 2014. ****Please cite both of these sources when
using this package.****


The power law PDF and error bars for the exponents are estimated using Mo's method; that is,
the error is taken as the range of exponent values found when the xmin and xmax boundaries
are varied by +-0.25 dex. This is a purposeful overestimate of the error, in the event that bootstrapping
gives errors that seem too low.

For more details, please read:
    [3] M.A. Sheikh, R.L. Weaver, K.A. Dahmen, Phys. Rev. Lett. 117 (2016).
    [4] M.A. Sheikh, Avalanches in Stars and at Finite Temperature, 2019.
    [5] N. Friedman, Avalanches in disordered systems (2013).
    
    
***QUICKSTART GUIDE:

    TO TEST WHICH DISTRIBUTION IS MOST LIKELY:
        1. Obtain the A^2 values from the ad() test on sizes and durations. Distributions with lowest A^2 are most consistent with the data.
        2. AND/OR use the LLR test contained in the llr_wrapper() function.
            If llr > 0 and p <= p_crit (for some chosen p_crit), then the first function is preferred.
            If llr < 0 and p <= p_crit (for some chosen p_crit), then the second function is preferred.
            Otherwise, neither is statistically preferred.
            I choose p_crit = 0.01 for my analysis, but this is a quite stringent boundary I've found.
            ***please note the LLR test only works when comparing two distributions created from the same number of data!!***


    TO OBTAIN ERRORBARS ON THE POWER LAW:
        1. Use out = bootstrap() to obtain histograms which explain the variance of all exponents.
        2. Obtain confidence intervals from the confidence_intervals() function. Default is 95% CI
            ex) med, hiCI, loCI = confidence_intervals(out[0]) gives median value med, 95%CI = (loCI, hiCI)
        3. Test the exponent relationships using tost(). Most stars pass tost with p < 0.1 when sig is set to, like, 0.2.
    
"""

#my packages
#from .j_powerlaw import j_powerlaw as jpl
from .get_ccdf_arr import get_ccdf_arr as ccdf
from .logbinning import logbinning
from .fit import fit
from .shapes import shapes


#python packages
import random
import mpmath as mp
from mpmath import gammainc
import numpy as np
import scipy
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import chi2
from scipy import optimize
import time
import multiprocessing as multi
import os
import numba

#vectorized operations
mylog = np.vectorize(mp.log)
myexp = np.vectorize(mp.exp)
mysqrt = np.vectorize(mp.sqrt)
myerfc = np.vectorize(mp.erfc)
myround = np.vectorize(round)
myfloat = np.vectorize(float)

arr = np.array

##TESTING SUITE

#find nearest value in an index. From StackOverflow.
@numba.njit
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#ASSUME DATA IS SORTED AND DATA[0] = XMIN AND DATA[-1] = XMAX
#Validated against https://doi.org/10.2478/s11600-013-0154-9 on 6-12-24, though the way they define the CDF goes from 1 to 0 and thus some algebra is required to get the same result as below.

#try the jited version
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

#find the power law index "exactly" using the zero of the derivative of the log likelihood function.
#Works more quickly and over a broader range than scipy.minimize version.
#ASSUME X IS SORTED AND IT GOES FROM XMIN TO XMAX INCLUSIVE

#Agrees with MLE solution to "Truncated Pareto" distribution from Table 1 of https://pearl.plymouth.ac.uk/bitstream/handle/10026.1/1571/2013Humphries337081phd.pdf?sequence=1
#TODO: find if we can replace root finding with https://github.com/Nicholaswogan/NumbaMinpack?
def find_pl_exact_sorted(x):
    ln = np.log
    xmin = x[0]
    xmax = x[-1]
    n = len(x)
    S = np.sum(np.log(x))
    
    #if not enough data, return a nan value
    if xmax/xmin < 1.5:
        return np.nan, -1e12
    
    #using function values only speeds up calculation
    def wrap(alpha):
        #test for alpha = 1
        if alpha == 1:
            return -ln(ln(xmax/xmin)) - S/n #equation from Deluca & Corrall 2013, equation 12.
        
        #large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
        test_xmin = np.log10(xmin)*(-alpha+1)
        if test_xmin > 100:
            return -1e12
        beta = -xmax**(-alpha+1) + xmin**(-alpha+1)
        gam = xmax**(-alpha+1)*ln(xmax) - xmin**(-alpha+1)*ln(xmin)
        
        #dgamdalpha = xmin**(-alpha+1)*(ln(xmin)**2)- xmax**(-alpha+1)*(ln(xmax)**2)
        f = n/(alpha - 1) - S - n*(gam/beta)
        #first derivative can be useful for more accurate convergence at cost of calculation speed.
        #df = -n/(alpha-1)**2 - n*(dgamdalpha/beta - (gam/beta)**2)
        
        return f#, df
    
    #if using first derivatives. Recommend not doing so for speed reasons.
    #alpha = scipy.optimize.root_scalar(wrap, bracket = (1,100), fprime = True).root

#    try:
    out = scipy.optimize.root_scalar(wrap, bracket = (1, 20))
#    except ValueError:
#        print('Value error!')
#        print(x)
#        return np.nan, -1e12
    alpha = out.root
    ll = pl_like(x,x[0],x[-1],alpha)[0]
    return alpha,ll

#wrapper for find_pl which does not assume x is sorted and is in the range of xmin and xmax.
def find_pl_exact(x,xmin,xmax = 1e6):
    x = arr(x)
    tmp = x[(x >= xmin)*(x <= xmax)]
    alpha,ll = find_pl_exact_sorted(np.sort(tmp))
    return alpha,ll

#used to estimate the pq value from D in the monte carlo xmin/xmax
def expfun(x,numterms = 5):
    val = 0
    for i in range(1,numterms+1):
        val = val + (-1)**(i-1)*np.exp(-2* i**2 * x**2)
        
    return 2*val

#From Clauset et al 2009, they test their method for determining xmin using a random variable sampled from
#a continuous, differentiable, piecewise pdf which follows exp(-alpha*x) for x < xmin and a power law for x > xmax. The inverse CDF shown here can be used to generate synthetic data.
def clauset_generate_test_data(datasize,xmin,alpha):
    x = np.random.rand(datasize)
    lam = (xmin/alpha)*(np.exp(alpha)-1)
    eta = xmin/(alpha-1)
    c_star = lam/(lam + eta)
    
    outs = np.zeros(len(x))
    
    for i in range(len(x)):
        if x[i] < c_star:
            outs[i] = xmin - (xmin/alpha)*np.log(np.exp(alpha) - (lam + eta)*alpha*x[i]/xmin)
        else:
            outs[i] = (eta/((lam + eta)*(1-x[i])))**(1/(alpha-1))*xmin
            
    return outs

#Test finding data with xmin and xmax, where power law scaling is noted between xmin and xmax and goes exponential elsewhere.
#Enforced continuity at boundaries of xmin and xmax.
@numba.njit
def generate_test_data_with_xmax(datasize,xmin,xmax,alpha):
    
    ln = np.log
    exp = np.exp
    x = np.random.rand(datasize)
    
    lam = xmin**(-alpha+1)*(np.exp(alpha) -1)/alpha
    eta = (xmin**(-alpha+1) - xmax**(-alpha + 1))/(alpha-1)
    gam = xmax**(-alpha + 1)/alpha
    
    co = 1/(lam + eta + gam)
    
    outs = np.zeros(len(x))
    
    c_star_lo = lam*co
    c_star_hi = (lam + eta)*co
    
    for i in range(len(x)):
        if x[i] < c_star_lo:
            outs[i] = (xmin/alpha)*ln(exp(alpha)/(exp(alpha) - (alpha/co)*xmin**(alpha-1)*x[i]))
        elif x[i] < c_star_hi:
            outs[i] = (xmin**(-alpha + 1) + (alpha-1)*(lam - x[i]/co))**(1/(-alpha+1))
        else:
            outs[i] = xmax - (xmax/alpha)*ln(1-((x[i]/co)-lam-eta)*xmax**(alpha-1)*alpha)
            
    return outs

#find the alpha using Brent's algorithm. This version is about 10x faster than the previous approach and gives the same answer, thanks to numba
#Gives same results as Scipy version which uses a c backend as implemented at https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/Zeros/brentq.c
#Fortran version that's slightly different is implemented in https://websites.pmc.ucsc.edu/~fnimmo/eart290c_17/NumericalRecipesinF77.pdf.
@numba.njit
def brent_findmin(x,blo = 1, bhi = 20, xtol = 1e-12, rtol = 8.881784197001252e-16, maxiter = 100):
    ln = np.log
    x = np.sort(x)
    xmin = x[0]
    xmax = x[-1]
    n = len(x)
    S = np.sum(np.log(x))
    def f(alpha):
        #test for alpha = 1
        if alpha == 1:
            return -ln(ln(xmax/xmin)) - S/n #equation from Deluca & Corrall 2013, equation 12.
        #large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
        test_xmin = np.log10(xmin)*(-alpha+1)
        if test_xmin > 100:
            return -1e12
        beta = -xmax**(-alpha+1) + xmin**(-alpha+1)
        gam = xmax**(-alpha+1)*ln(xmax) - xmin**(-alpha+1)*ln(xmin)
        y = n/(alpha - 1) - S - n*(gam/beta)
        
        return y
    
    #hold previous, current, and blk (?) values
    xpre = blo #previous estimate of the root
    xcur = bhi #current estimate of the root
    #xblk = 0 #holds value of x (?)
    fpre = f(xpre)
    fcur = f(xcur)
    #fblk = 0 #hold value of f(x) (?)

    #s values
    spre = 0 #previous step size
    scur = 0 #current step size
    sbis = 0 #bisect
    stry = 0
    
    #d values
    dpre = 0
    dblk = 0
    
    delta = 0 #hold the value of the error
    
    #edge case calculation
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur
    tol1 = -1
    
    #main loop
    for i in range(maxiter):
        #if fpre and fcur are both not zero and fpre has a different sign from fcur
        if ((fpre != 0)*(fcur != 0)*(np.sign(fpre) != np.sign(fcur))):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre #put xpre and fpre into xblk and fblk, set spre = scur = xcur - xpre
            
        #if fblk is less than fcur, then move the bracket to (xpre, xblk)
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre
            
            fpre = fcur
            fcur = fblk
            fblk = fpre
        
        #check the bounds
        delta = 0.5*(xtol + rtol*abs(xcur))
        sbis = 0.5*(xblk - xcur)
        if ((fcur == 0) + (abs(sbis) < delta)):
            #print('Root found in %d iterations.' % i)
            return xcur
        
        if (abs(spre) > delta)*(abs(fcur) < abs(fpre)):
            if xpre == xblk:
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre))
            #short step
            if (2*abs(stry) < abs(spre)) * (2*abs(stry) < 3*abs(sbis) - delta):
                spre = scur
                scur = stry
            #otherwise bisect
            else:
                spre = sbis
                scur = sbis
        else:
            #otherwise bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur = xcur + scur #step xcur by scur
        else:
            xcur = xcur + np.sign(sbis)*delta
        
        fcur = f(xcur) #another function call
    
    #print('Max iters achieved.')
    return xcur

#find the true p value (as compared to pq). See Deluca & Corrall 2013 (https://doi.org/10.2478/s11600-013-0154-9).
#If true p > 0.15 (or maybe p > 0.2), then the power law fit is good.
@numba.njit
def find_true_p(x,xmin,xmax,runs = 150, dfun = find_d_sorted):
    
    tot = 0
    p = 1
    sigp = 0
    x = np.sort(x)
    x = x[(x >= xmin)*(x <= xmax)]
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
    
#use a monte carlo approach of finding xmin, xmax, and alpha using KS statistic.
#Set the default pcrit to 0.45, since that corresponds to a true p of approx 0.2 and gives very good reliability in testing.

#Cannot njit because you can't assign a function to a variable name in nopython mode.
#@numba.njit
def find_pl_montecarlo(data, runs = 2000, pqcrit = 0.35, pcrit = 0.2, pruns = 150, dist_type = 'KS'):
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
    if dist_type == 'KS':
        dfun = find_d_sorted
    elif dist_type == 'AD':
        dfun = find_ad_sorted
    else:
        print('Error. Please input a valid distance type (either KS or AD)')
        return 1, 0, 0, 0, 0
    
    #As of 6-12-24, use an updated method that searches for any with pq > 0.35 (gives p approx 0.15 or so) then selects the run that maximizes xmax/xmin
    attempted_xmins = np.ones(runs)
    attempted_xmaxs = np.ones(runs)
    attempted_alphas = np.ones(runs)
    attempted_ns = np.ones(runs)
    attempted_pqs = np.ones(runs)
    attempted_ds = 1e12*np.ones(runs)
    log_xmin = np.log(data[0])
    log_xmax = np.log(data[-1])
    log_range = log_xmax-log_xmin
    for i in range(runs):
        #if (np.mod(i,1000) == 0):
        #    print(i)
        trial_xmin = np.exp(log_xmin + log_range*np.random.rand())
        trial_xmax = np.exp(log_xmin + log_range*np.random.rand())
        trial_xmin_idx = find_nearest_idx(data,trial_xmin)
        trial_xmax_idx = find_nearest_idx(data,trial_xmax)
        trial_xmin = data[trial_xmin_idx]
        trial_xmax = data[trial_xmax_idx]
        while (trial_xmax_idx - trial_xmin_idx < 10) + (trial_xmax < 2*trial_xmin):        
            trial_xmin = np.exp(log_xmin + log_range*np.random.rand())
            trial_xmax = np.exp(log_xmin + log_range*np.random.rand())
            trial_xmin_idx = find_nearest_idx(data,trial_xmin)
            trial_xmax_idx = find_nearest_idx(data,trial_xmax)
            trial_xmin = data[trial_xmin_idx]
            trial_xmax = data[trial_xmax_idx]
        trimmed = data[trial_xmin_idx:trial_xmax_idx+1]
        alpha_hat = brent_findmin(trimmed)
        tmpd = 1e12
        if alpha_hat > 1:
            attempted_ds[i] = dfun(trimmed,alpha_hat)
            tmpd = find_d_sorted(trimmed,alpha_hat)
        n = len(trimmed)
        #Function is Equation 29 in Deluca & Corrall 2013 (https://doi.org/10.2478/s11600-013-0154-9)
        #gives a "fake p" that correlates with the real p
        attempted_pqs[i] = expfun(tmpd*np.sqrt(n) + 0.12*tmpd + 0.11*tmpd/np.sqrt(n))
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
    print(len(possible_pqs))
    
    possible_ps = np.zeros(len(possible_pqs))
    for i in range(len(possible_pqs)):
        xmin = possible_xmins[i]
        xmax = possible_xmaxs[i]
        xmin_idx = find_nearest_idx(data,xmin)
        xmax_idx = find_nearest_idx(data,xmax)
        trimmed = data[xmin_idx:xmax_idx + 1]
        tmp = find_true_p(trimmed,xmin,xmax, runs = pruns, dfun = dfun)[0]
        possible_ps[i] = tmp
    
    #only examine runs where the p-value is greater than the pcrit (default 0.2)
    idxs2 = np.where(possible_ps > pcrit)[0]
    possible_xmins2 = possible_xmins[idxs2]
    possible_xmaxs2 = possible_xmaxs[idxs2]
    possible_alphas2 = possible_alphas[idxs2]
    possible_pqs2 = possible_pqs[idxs2]
    possible_ps2 = possible_ps[idxs2]
    print(len(possible_ps2))
    
    minidx = np.argmax(possible_xmaxs2/possible_xmins2)
    
    xmin = possible_xmins2[minidx]
    xmax = possible_xmaxs2[minidx]
    alpha = possible_alphas2[minidx]
    
    #the "false" p-value from equation 29 of (https://doi.org/10.2478/s11600-013-0154-9).
    pq = possible_pqs2[minidx]
    p = possible_ps2[minidx]
    
    return alpha, xmin, xmax, pq, p
    
    
    
    

##LIKELIHOOD FUNCTIONS

def pl_like(x,xmin,xmax,alpha):
    ll = 0
    x = np.array(x)
    x = x[(x >= xmin)* (x <= xmax)]
    X = xmax/xmin
    dist = np.log(((alpha-1)/xmin)*(1/(1-X**(1-alpha)))*(x/xmin)**(-alpha))
    ll = sum(dist)
    return ll, dist

#fast version of pl_like
#ENSURE INPUT X is geq xmin and leq xmax!
def pl_like_fast(x,xmin,xmax,alpha):
    ll = 0
    X = xmax/xmin
    dist = np.log(((alpha-1)/xmin)*(1/(1-X**(1-alpha)))*(x/xmin)**(-alpha))
    ll = sum(dist)
    return ll, dist

#exponential log likelihood. Matches with powerlaw library!
def exp_like(x,xmin,xmax,lam):
    if lam <=0:
        return -1e-12, np.zeros(len(x))
    x = np.array(x)
    x = x[(x >= xmin)*(x <= xmax)]
    x = x - xmin #subtract minimum to turn it into a normalized PDF (Clauset et al 2009)
    #MLE estimate for lambda says lambda_opt = 1/np.mean(x)
    #lam = 1/np.mean(x)
    N = len(x)
    dist = np.log(lam*np.exp(-lam*x))
    ll = N*np.log(lam) - N*lam*np.mean(x)
    return ll, dist

def lognormal_like(x, xmin, xmax, mu, sigma):
    
    ##  NOTE: the corresponding log-likelihood function used by the powerlaw() library
    #   does not appropriately limit the boundaries for mu or sigma. Mu can, in
    #   principle, be any value. Negative values of mu might be expected if the generative process is from
    #   multiplication of many positive random variables, for instance. This is possible in our system and AGNs, so
    #   we should not limit mu or sigma. While AGN and stars are very different systems, the MHD equations
    #   should still apply in both cases, albeit in different limits.
    
    #catch the illegal values of mu and sigma
    #if sigma <= 0 or mu < log(xmin):
    #    return -1e12, np.zeros(len(x))
    x = np.array(x)
    x = x[(x >= xmin)*(x <= xmax)]
    n = len(x)
    pi = mp.pi
    #log likelihood is the sum of the log of the likelihoods for each point. Likelihood function is just pdf(x) for all x.
    #mpmath is used because it has higher accuracy than scipy
    dist = -mylog(x)-((mylog(x)-mu)**2/(2*sigma**2)) + 0.5*mylog(2/(pi*sigma**2))- mylog(myerfc((mylog(xmin)-mu)/(mysqrt(2)*sigma)))
    ll = float(sum(dist))
    dist = myfloat(dist) #convert to float
    return ll, dist

#get the truncated power law likelihood function. Restricts values to be alpha > 1 and lambda > 0. Matches with powerlaw() library!
def tpl_like(x,xmin,xmax,alpha,lam):
    if alpha <= 1 or lam <= 0 or len(x) <= 5:
        return -1e12, np.zeros(len(x))
    x = x[(x >= xmin)*(x <= xmax)]
    dist = (1-alpha)*np.log(lam) - alpha*np.log(x) - lam*x - mylog(gammainc(1-alpha,lam*xmin))
    ll = float(sum(dist))
    dist = myfloat(dist)
    return ll, dist

#get the truncated power law likelihood function. Restricts values to be alpha > 1 and lambda > 0.
#ENSURE THE INPUT X IS AN ARRAY BETWEEN XMIN AND XMAX!!
def tpl_like_fast(x,xmin,alpha,lam):
    if alpha <= 1 or lam <= 0 or len(x) <= 5:
        return -1e12, np.zeros(len(x))
    dist = (1-alpha)*np.log(lam) - alpha*np.log(x) - lam*x - mylog(gammainc(1-alpha,lam*xmin))
    #dist = (1-alpha)*np.log(lam) - alpha*np.log(x) - lam*x - expn(alpha-1,lam*xmin)/(lam*xmin) #this code *would* work, and would be much faster, but scipy does not allow for float inputs to Expn!
    ll = float(sum(dist))
    dist = myfloat(dist)
    return ll, dist

#v2 use minimize_scalar to be about 3-4x faster than nelder-mead.
def find_pl(x,xmin,xmax = 1e6):
    x = np.array(x)
    xc = x[(x > xmin)*(x < xmax)]
    mymean = lambda a: -pl_like(xc,xmin,xmax,a)[0]
    #myfit = optimize.minimize(mymean,2,method = 'Nelder-Mead', bounds = [(1,1e6)])
    myfit = optimize.minimize_scalar(mymean, bounds = (1,30))
    ll = -myfit.fun
    alpha = myfit.x
    return alpha,ll

##MLE FITS
#find the exponents for a powerlaw between xmin and xmax, with errorbars
#ENSURE X IS BETWEEN XMIN AND XMAX!

#v2 use minimize scalar to be about 3-4x faster than nelder-mead.
def find_pl_fast(x,xmin,xmax = 1e6):
    if len(x) == 1:
        return np.nan, -1e12
    mymean = lambda a: -pl_like_fast(x,xmin,xmax,a)[0]
    #myfit = optimize.minimize(mymean,2,method = 'Nelder-Mead',bounds = [(1,1e6)])
    myfit = optimize.minimize_scalar(mymean, bounds = (1,30))
    ll = -myfit.fun
    alpha = myfit.x
    return alpha,ll


#find the truncated power law parameters. Matches with the output from powerlaw()
def find_tpl(x,xmin,xmax = 1e6):
    x = x[(x >= xmin)*(x <= xmax)]
    #initial_guess = [1 + len(x)/sum(np.log(x/xmin)), 1/np.mean(x)]
    initial_guess = [2,1/np.mean(x)]
    mymean = lambda par: -tpl_like(x,xmin,xmax,par[0],par[1])[0]
    opt_results = optimize.minimize(mymean,initial_guess,method = 'Nelder-Mead')
    #output value
    alpha = opt_results.x[0]
    lam = opt_results.x[1]
    
    #Maximum likelihood
    ll = -opt_results.fun
    return alpha,lam,ll

#fast version of find_tpl.
#ENSURE THE INPUT IS LIMITED TO BE BETWEEN XMIN AND XMAX!!
def find_tpl_fast(x,xmin,xmax = 1e6):
    #initial_guess = [1 + len(x)/sum(np.log(x/xmin)), 1/np.mean(x)]
    initial_guess = [2,1/np.mean(x)]
    mymean = lambda par: -tpl_like_fast(x,xmin,par[0],par[1])[0]
    opt_results = optimize.minimize(mymean,initial_guess,method = 'Nelder-Mead')
    #output value
    alpha = opt_results.x[0]
    lam = opt_results.x[1]
    
    #Maximum likelihood
    ll = -opt_results.fun
    return alpha,lam,ll

#find the best-matching exponential distribution using MLE. Matches with powerlaw()
def find_exp(x,xmin,xmax = 1e6):
    x = np.array(x)
    initial_guess = [1/np.mean(x[(x>=xmin)*(x<=xmax)])]
    mymean = lambda par: -exp_like(x,xmin,xmax,par[0])[0]
    opt_results = optimize.minimize(mymean,initial_guess,method = 'Nelder-Mead')
    lam = opt_results.x[0]
    ll = -opt_results.fun
    return lam,ll

def find_lognormal(x,xmin,xmax = 1e6):
    x = np.array(x)
    logx = np.log(x[(x >= xmin)*(x <= xmax)])
    initial_guess = [np.mean(logx),np.std(logx)]
    mymean = lambda par: -lognormal_like(x,xmin,xmax,par[0],par[1])[0]
    opt_results = optimize.minimize(mymean,initial_guess,method = 'Nelder-Mead')
    mu = opt_results.x[0]
    sigma = opt_results.x[1]
    ll = -opt_results.fun
    return mu,sigma,ll

##LOG LIKELIHOOD RATIO COMPARISON.
#The truncated power law is nested within the power law, so set "nested" to true when comparing those two.
def llr(dist1,dist2,nested = False):
    length = len(dist1)
    llr = sum(dist1 - dist2)
    var = np.var(dist1-dist2)
    if nested:
        p = 1 - chi2.cdf(abs(2*llr), 1)
    else:
        p = float(myerfc(abs(llr)/np.sqrt(2*length*var))) #obtain p-value from distribution compare by using error function (assumes the test statistic is normally distributed with mean value llr and variance var)
    return llr, p

"""
Wrapper function for llr. Finds the llr and p value for two distributions. Positive value and p <= p_thresh prefers the first function over the second.
Negative number and p <= p_thresh prefers the second distribution. If p > p_thresh, then neither is significantly preferred between the two.

Recommend p_thresh = 0.01 to reproduce results in Kepler manual analysis.

Variables:
    --x: vector of (size/dur/vmax) data.
    --xmin: the minimum of the scaling regime
    --xmax: maximum of the scaling regime
    --dists:
        An array of two strings, telling which two distributions will be tested against one another.

"""
def llr_wrap(x,xmin,xmax, totest = ['power_law','exponential']):
    #check if inputs for distributions are legal
    legal = ['power_law','exponential','lognormal','truncated_power_law']
    if not hasattr(totest, '__iter__'):
        print('Please give dists as a list/array/other iterable of two strings!')
        print('Legal choices are power_law, exponential, truncated_power_law, lognormal')
        return
    if not all(i in legal for i in totest):
        print('Please give valid distributions!')
        print('Legal choices are power_law, exponential, truncated_power_law, lognormal')
        return
    if not all(type(i) is str for i in totest):
        print('Distributions should be given as strings!')
        print('Legal choices are power_law, exponential, truncated_power_law, lognormal')
        return
    if len(totest) != 2:
        print('Please give exactly two strings for dist!')
        return
    if totest[0] == totest[1]:
        print('Please give different strings!')
    
    #if the inputs are 'power_law' and 'truncated_power_law', then the inputs are nested versions of one another.
    if len(set(totest) - set(['power_law','truncated_power_law'])) == 0:
        nested = True
    else:
        nested = False
    
    #Get the right functions to compare
    findfuns = [None]*2 #functions to get the optimal values
    llrfuns = [None]*2
    opts = [None]*2
    dists = [None]*2
    for i in range(len(totest)):
        if totest[i] == 'power_law':
            #print('pl')
            findfuns[i] = find_pl
            llrfuns[i] = pl_like
        if totest[i] == 'truncated_power_law':
            #print('tpl')
            findfuns[i] = find_tpl
            llrfuns[i] = tpl_like
        if totest[i] == 'exponential':
            #print('exp')
            findfuns[i] = find_exp
            llrfuns[i] = exp_like
        if totest[i] == 'lognormal':
            #print('ln')
            findfuns[i] = find_lognormal
            llrfuns[i] = lognormal_like
        opts[i] = findfuns[i](x,xmin,xmax)[:-1]
        dists[i] = llrfuns[i](x,xmin,xmax,*opts[i])[-1]

    #print(dists)
    #now that the dist lists are populated, get the optimal values using MLE
    ll,p = llr(dists[0],dists[1],nested = nested)
    return ll,p



#from powerlaw library. Edited to allow for an entire vector to be input at once.
def lognormal_gen(x,xmin,xmax,mu,sigma):
    from numpy import exp, sqrt, log, frompyfunc
    from mpmath import erf, erfinv
    #This is a long, complicated function broken into parts.
    #We use mpmath to maintain numerical accuracy as we run through
    #erf and erfinv, until we get to more sane numbers. Thanks to
    #Wolfram Alpha for producing the appropriate inverse of the CCDF
    #for me, which is what we need to calculate these things.
    erfinv = frompyfunc(erfinv,1,1)
    Q = erf( ( log(xmin) - mu ) / (sqrt(2)*sigma))
    Q = Q*x - x + 1.0
    Q = myfloat(erfinv(Q))
    return exp(mu + sqrt(2)*sigma*Q)

#produces a power law between xmin and xmax when supplied with a vector of random variables with elements 0 < r < 1
@numba.njit
def pl_gen(datalen,xmin,xmax,alpha):
    x = np.random.rand(datalen)
    X = xmax/xmin
    out = (1-x*(1-X**(1-alpha)))**(1/(1-alpha))*xmin
    return out

#from powerlaw() library. Gives a truncated power law with given alpha and lambda.
def tpl_gen(r,xmin,alpha,Lambda):
    def helper(r):
        from numpy import log
        from numpy.random import rand
        st = time.time()
        dt = time.time() - st
        while dt <= 0.1: #spend at most 0.1 seconds per data point
            dt = time.time() - st
            x = xmin - (1/Lambda) * log(1-r)
            p = ( x/xmin )**-alpha
            if rand()<p:
                return x
            r = rand()
        #If spent too much time, then return nan
        return np.nan
    from numpy import array
    return array(list(map(helper, r)))

#produces an exponentially distributed PDF with a given xmin.
def exp_gen(x,xmin,Lambda):
    out = -np.log(1-x)/Lambda
    out = out + xmin
    return out

#bootstrap by picking random subset of avalanches, then restricting analysis to
#smin < s < smax and dmin < d < dmax where smin/smax and dmin/dmax are varied
#*independently* of one another.
#Independent variation is considered so any effect of cross correlation introduced
#by deriving smin/smax from dmin/dmax is not considered; i.e. it errs on the side of
#caution by making error bars larger.

"""
4:58 5:01 5:55 5:55 5:59
Testing results:
    --snz gives the same errorbar as linear fit on log-log plot for scaling stars with more than 100
        events in the scaling regime. The mean value converges to the value obtained by a linear fit.
        The error for bootstrapping is larger when there fewer events, which makes sense.
    --There are times when alpha is close to 1, in which case (tau-1)/(alpha-1) blows up to nonphysical
        parameters. When plotting the histogram, only do so with indices where values (tau-1)/(alpha-1) < 10 or so
    --Since (tau-1)/(alpha-1) and snz are decided on each index for the same set of avalanches, the distribution of
        scaling relationship (tau-1)/(alpha-1) - snz can be obtained simply by subtracting the output vectors
        from one another.

"""

#get a new random xmin_star and xmax_star which varies over -dex to dex in log scale and ensures xmin < xmax.
#note that ensuring xmin > xmax makes the "random" value of xmax covariant with xmin.
def logrand(xmin,xmax,dex):
    #if dex is zero, xmin and xmax do not change.
    if dex == 0:
        return xmin,xmax
    logxmin = np.log10(xmin)
    logxmax = np.log10(xmax)
    
    logxmin_star = np.random.uniform(-dex + logxmin, dex + logxmin) #get log(xmin) to vary between -dex and +dex
    #ensure smax > smin while keeping sampling uniform
    logxmax_star = np.random.uniform(max([-dex + logxmax, logxmin_star]), dex + logxmax) #log(xmax_star) varies between -dex and dex if logxmin_star < -dex + logsmax_star, otherwise log(xmax_star) varies between logxmin_star and +dex
    
    xmin_star = 10**logxmin_star
    xmax_star = 10**logxmax_star
    return xmin_star, xmax_star

#find ymin and ymax from an interpolated function over logx,logy. Default to 50 log bins.
def loginterp(x,y,xmin,xmax, bins = 50):
    bx,by,_ = logbinning(x,y,bins)
    
    logbx = np.log10(bx)
    logby = np.log10(by)
    
    myinterp = interp1d(logbx,logby)
    lo = max([np.log10(xmin),min(logbx)])
    hi = min([np.log10(xmax),max(logbx)])
    
    logymin = myinterp(lo)
    logymax = myinterp(hi)
    
    ymin = 10**logymin
    ymax = 10**logymax
    return ymin,ymax

#using logbinned data to speed up calculations.
def binned_interp(bx,myinterp,xmin,xmax):
    logbx = np.log10(bx)
    lo = max([np.log10(xmin),min(logbx)])
    hi = min([np.log10(xmax),max(logbx)])
    
    logymin = myinterp(lo)
    logymax = myinterp(hi)
    
    ymin = 10**logymin
    ymax = 10**logymax
    return ymin,ymax
    

#bootstrapping core. From an input list of avalanche s,d,smin,smax,etc, estimate a single run of exponents.
def bootstrap_core(s,d,smin, smax, dmin, dmax,vm, vmin, vmax,logs,logd,logvm, fun, dex, ctr_max,myinterp):
    
    length = len(s)
    nums = 0
    numd = 0
    numv = 0
    ctr = 0
    
    #initialize variable values if ctr exceeds ctr_max and bootstrap needs to exit.
    tau = np.nan
    alpha = np.nan
    mu = np.nan
    
    snz = np.nan
    sp = np.nan
    pnz = np.nan
    
    sdlhs = np.nan #(tau-1)/(alpha-1)
    svlhs = np.nan #(tau-1)/(mu-1)
    dvlhs = np.nan #(alpha-1)/(mu-1)

    #if vm is not input, vmin and vmax should stay equal to 1.    
    vmin_star = 1
    vmax_star = 1
    
    
    #try to get enough events to bootstrap over (at least 1.)
    while (ctr <= ctr_max)*((nums < 1) + (numd < 1) + (numv < 1)):    
        idxs = np.random.randint(0,length,length) #get indexes of avalanches to sample
        smin_star,smax_star = logrand(smin,smax,dex) #get random smin and smax, ensuring smax > smin    
        dmin_star,dmax_star = logrand(dmin,dmax,dex) #get random dmin and dmax, ensuring dmax > dmin
        
        sc = s[idxs] #get list of sampled avalanche size, duration, velocity and their logs
        dc = d[idxs]
        vmc = vm[idxs]    
        logsc = logs[idxs]
        logdc = logd[idxs]    
        logvmc = logvm[idxs]
        
        #if vmin and vmax are something other than 1, estimate vm statistics.
        if (vmin != 1)*(vmax != 1):
            vmin_star,vmax_star = logrand(vmin,vmax,dex) #get random vmin and vmax, ensuring vmax > vmin        
            #vmin_star,vmax_star = binned_interp(sc,myinterp,smin_star,smax_star)
            
        nums = sum((sc >= smin_star)*(sc <= smax_star))
        numd = sum((dc >= dmin_star)*(dc <= dmax_star))
        numv = sum((vmc >= vmin_star)*(vmc <= vmax_star))
        ctr = ctr + 1
        
    #if the loop exited because ctr == ctr_max, return nans.
    if ctr >= ctr_max:
        return tau,alpha,mu, sdlhs,svlhs,dvlhs, snz,sp,pnz

    #otherwise, continue on.
    scc = sc[(sc >= smin_star)*(sc <= smax_star)]
    dcc = dc[(dc >= dmin_star)*(dc <= dmax_star)]
    vmcc = vmc[(vmc >= vmin_star)*(vmc <= vmax_star)]
    
    #all bounds are chosen wrt smin to ensure there is the same number of events in each array. Known source of potential value dependence.
    logscc = logsc[(sc >= smin_star)*(sc <= smax_star)]
    logdcc = logdc[(sc >= smin_star)*(sc <= smax_star)]
    logvmcc = logvmc[(sc >= smin_star)*(sc <= smax_star)]
    
    #old if statement. Was required to check if xmin < xmax for all x and that len(xcc) >= 3 for all of s, d, and vm, and that all values are different.
    #using find_pl_exact, these checks are not required.
    #ctr < ctr_max and (cursmin >= cursmax or curdmin >= curdmax or len(scc) <= 3 or len(dcc) <= 3 or len(vmcc) <= 3 or all(scc == scc[0]*np.ones(len(scc))) or all(dcc == dcc[0]*np.ones(len(dcc))) or all(vmcc == vmcc[0]*np.ones(len(vmcc))))

    #find tau and alpha
    tau = fun(scc,smin_star,smax_star)[0]
    alpha = fun(dcc, dmin_star,dmax_star)[0]
    
    #find snz and (tau-1)/(alpha-1)
    snz = scipy.stats.linregress(logscc,logdcc).slope
    sdlhs = (tau-1)/(alpha-1)
    
    #if vm is given, also calculate velocity statistics.
    if (vmin != 1)*(vmax != 1):
        mu = fun(vmcc,vmin_star,vmax_star)[0]
        sp = scipy.stats.linregress(logscc,logvmcc).slope
        pnz = scipy.stats.linregress(logdcc,logvmcc).slope
        svlhs = (tau-1)/(mu-1)
        dvlhs = (alpha-1)/(mu-1)
    
    
    return tau,alpha,mu, sdlhs,svlhs,dvlhs, snz,sp,pnz
    
#multiprocessing helper function
def worker(index, ins, taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs):
    return index, bootstrap_core(*ins)

#v2 of bootstrap, updated Feb 27, 2024. Written to take advantage of various programming fundamentals improvements Jordan learned since the original bootstrap was written.
def bootstrap2(s,d, smin, smax, dmin, dmax, vm = None, num_runs = 10000, mytype = 'power_law_exact', dex = 0.25, ctr_max = 10, min_events = 10, parallel = False):
    
    #ctr_max is the max number of times to try reshuffling before skipping a particular run.
    taus = np.array([np.nan]*num_runs)
    alphas = np.array([np.nan]*num_runs)
    sdlhss = np.array([np.nan]*num_runs)

    mus = np.array([np.nan]*num_runs) #exponent on vmax CCDF
    svlhss = np.array([np.nan]*num_runs) #relationship (tau-1)/(mu-1)
    dvlhss = np.array([np.nan]*num_runs) #relationship (alpha-1)/(mu-1)
    
    snzs = np.array([np.nan]*num_runs)
    sps = np.array([np.nan]*num_runs) #vmax vs size power law exponent
    pnzs = np.array([np.nan]*num_runs) #vmax vs duration power law exponent (rho)/(nu*z)

    num_avs = len(s)
    
    #hold log of s and d so it doesnt have to be recalculated every run
    logs = np.log10(s)
    logd = np.log10(d)
    if vm is None:
        vm = np.ones(num_avs)
        vmin = 1
        vmax = 1
        myinterp = None
    else:
        vmin,vmax = loginterp(s,vm,smin,smax, bins = 50) #use for independent vm dex selection, which leads to unreasonably large error bars since vm should have such a small scaling regime most of the time.
        
        bs,bv,_ = logbinning(s,vm,50)
        myinterp = interp1d(np.log10(bs),np.log10(bv))
        #vmin,vmax = binned_interp(bs,myinterp,smin,smax)
        

    logvm = np.log10(vm)
    
    
    #if there are fewer than min_events number of events in s or d, return.
    if sum((s >= smin)*(s <= smax)) <= min_events or sum((d >= dmin)*(d <= dmax)) <= min_events:
        print("Not enough events. Returning.")
        return taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs
    
    if mytype == 'power_law':
        fun = find_pl_fast #set fun to be the power_law() function
    elif mytype == 'power_law_exact':
        fun = find_pl_exact #get the exact solution. Preferred.
    elif mytype == 'truncated_power_law':
        fun = find_tpl #set fun to be the find_tpl() function instead. Not tested.
    else:
        print('Wrong option for function, please choose any of power_law, power_law_exact, or truncated_power_law. Returning.')
        return taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs
    
    #do the bootstrapping (serial). A bit slower than the original bootstrapping approach.
    if not parallel:
        for i in range(num_runs):
            if i % 1000 == 0:
                print(i)
            taus[i],alphas[i],mus[i],sdlhss[i],svlhss[i],dvlhss[i],snzs[i],sps[i],pnzs[i] = bootstrap_core(s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,fun,dex,ctr_max, myinterp)
            
        return taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs
    #otherwise, it's parallel.        
    pool = multi.Pool(os.cpu_count())
    ins = (s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,fun,dex,ctr_max, myinterp)
    
    #using apply_async (2x speedup on 6-core computer)
    results = [pool.apply_async(worker, args=(i, ins, taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs)) for i in range(num_runs)]
    
    for result in results:
        i, val = result.get()
        taus[i] = val[0]
        alphas[i] = val[1]
        mus[i] = val[2]
        sdlhss[i] = val[3]
        svlhss[i] = val[4]
        dvlhss[i] = val[5]
        snzs[i] = val[6]
        sps[i] = val[7]
        pnzs[i] = val[8]
    
    # Close the pool to free resources
    pool.close()
    pool.join()

    #return values    
    return taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs

#the core of the BCa correction given theta_hat and jackknifed samples theta_jk. Validated against scipy.stats.bootstrap.
def bca_core(theta_hat,theta_jk, bootstrap_estimates, alpha):
    
    z0 = scipy.stats.norm.ppf((np.sum(bootstrap_estimates < theta_hat) + 0.5) / (len(bootstrap_estimates) + 1))

    theta_jk_dot = np.mean(theta_jk)
    
    #SciPy implementation of BCa.
    #NOTE: the numdat term drops out algebraically, so I am unsure why it's there in the SciPy implementation (shown below)
    #top = np.sum((theta_jk_dot - theta_jk)**3/numdat**3)
    #bot = np.sum((theta_jk_dot - theta_jk)**2/numdat**2)
    #a_hat = top/(6*bot**(3/2))
    

    #Equation 14.15 on pp. 186 of "An Introduction to the Bootstrap" by Efron and Tibshirani implementation
    #validated against test data given in the book. For test data a = [48, 36, 20, 29, 42, 42, 20, 42, 22, 41, 45, 14,6, 0, 33, 28, 34, 4, 32, 24, 47, 41, 24, 26, 30,41]
    #given on pp. 180, a_hat should equal 0.61 (given in pp. 186). Formula below gives correct answer.
    #Also checked bca_core against SciPy.
    top = np.sum((theta_jk_dot - theta_jk)**3)
    bot = np.sum((theta_jk_dot - theta_jk)**2)
    a_hat = top/(6*bot**(3/2))

    
    #confidence interval assuming normality
    z_alpha = scipy.stats.norm.ppf(alpha / 2)
    z1_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
    
    # Correct z-score with BCa adjustment
    z_bca1 = z0 + (z0 + z_alpha) / (1 - a_hat * (z0 + z_alpha))
    z_bca2 = z0 + (z0 + z1_alpha) / (1 - a_hat * (z0 + z1_alpha))

    #Obtain the lower and upper alpha values
    alpha_1 = scipy.special.ndtr(z_bca1)
    alpha_2 = scipy.special.ndtr(z_bca2)
    
    #obtain the confidence interval from the bootstrap estimates. Ignore nans.
    ci_lower = np.nanpercentile(bootstrap_estimates,alpha_1*100)
    ci_upper = np.nanpercentile(bootstrap_estimates,alpha_2*100)
    return theta_hat, ci_lower, ci_upper
    
#Find the bias corrected and accelerated (BCa) confidence intervals for power law exponents from the bootstrapping function.
#Returns the true value of the statistic and its lower and upper confidence intervals given output from bootstrap function.
#Method validated against scipy.stats.bootstrap with BCa selected.
def bca_pl(x, xmin,xmax, bootstrap_estimates, ci = 0.95):    
    alpha = 1-ci #get alpha (e.g. confidence interval of 95% means alpha = 0.05)
    
    #compute the "true" value of theta
    xc = x[(x >= xmin)*(x <= xmax)]
    theta_hat = find_pl_exact(xc,xmin,xmax)[0]    
    
    #Jackknife estimation of acceleration (from Scipy)
    numdat = len(xc)
    theta_jk = np.zeros(numdat)
    for i in range(numdat):
        theta_jk[i] = find_pl_exact(np.delete(xc,i),xmin,xmax)[0]

    #calculate    
    return bca_core(theta_hat,theta_jk, bootstrap_estimates, alpha)

#compute the BCA confidence intervals for combined statistics. i.e. like (tau-1)/(alpha-1)
def bca_lhs(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates, ci = 0.95):
    alpha = 1-ci
    
    xc = x[(x >= xmin)*(x <= xmax)]
    yc = y[(y >= ymin)*(y <= ymax)]
    
    thetax_hat = find_pl_exact(xc,xmin,xmax)[0]
    thetay_hat = find_pl_exact(yc,ymin,ymax)[0]

    #get the mean value of (tau-1)/(alpha-1)
    theta_hat = (thetax_hat - 1)/(thetay_hat - 1)
    
    nx = len(xc)
    ny = len(yc)
    
    #The real way would be to jackknife over avalanches.
    #That is, remove avalanches (1) one by one (remove same index i,j), (2) compute xc and yc from xmin, xmax, ymin, ymax, (3) compute tau, alpha from the subsample
    
    #assuming samples of x and y are independent (they're not! But the resampling in bootstrapping assumes s,d,etc are independent)
    ""
    #jackknife over thetax
    thetax_jk = np.zeros(nx)
    for i in range(nx):
        thetax_jk[i] = find_pl_exact(np.delete(xc,i),xmin,xmax)[0]
    
    #jackknife over thetay
    thetay_jk = np.zeros(ny)
    for i in range(ny):
        thetay_jk[i] = find_pl_exact(np.delete(yc,i),ymin,ymax)[0]
        
    #get the total jackknife (??)
    theta_jk = np.zeros(nx*ny)
    for i in range(nx):
        for j in range(ny):
            theta_jk[int(j*nx + i)] = (thetax_jk[i] - 1)/(thetay_jk[j] - 1) #compute (tau-1)/(alpha-1) for all jackknifed nx and ny, taking advantage of independence of x and y
        
    ""
    return bca_core(theta_hat,theta_jk, bootstrap_estimates, alpha)

#Do BCa on fitted functions (i.e. like snz)
def bca_fit(x,y,xmin,xmax,bootstrap_estimates,ci = 0.95):
    alpha = 1-ci
    
    
    xc = x[(x>= xmin)*(x <= xmax)]
    yc = y[(x >= xmin)*(x <= xmax)]
    
    logxc = np.log10(xc)
    logyc = np.log10(yc)

    theta_hat = scipy.stats.linregress(logxc,logyc).slope
    
    nx = len(xc)
    theta_jk = np.zeros(nx)
    for i in range(nx):
        theta_jk[i] = scipy.stats.linregress(np.delete(logxc,i),np.delete(logyc,i)).slope
        
    return bca_core(theta_hat,theta_jk,bootstrap_estimates,alpha)

#BCa to determine the exponent relationship confidence intervals, i.e. (tau-1)/(alpha-1) = snz
#HOW TO USE: pass in bca_rel(s,d,smin,smax,dmin,dmax,lhss,snzs, ci = 0.95) to get bootstrapping.
def bca_rel(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates,bootstrap_estimates2, ci = 0.95):
    alpha = 1 - ci
    
    xc = x[(x>= xmin)*(x <= xmax)]
    yc = y[(y >= ymin)*(y <= ymax)]
    
    logxc = np.log10(xc)
    logyc = np.log10(y[(x >= xmin)*(x <= xmax)])

    fit_hat = scipy.stats.linregress(logxc,logyc).slope
    thetax_hat = find_pl_exact(xc,xmin,xmax)[0]
    thetay_hat = find_pl_exact(yc,ymin,ymax)[0]
    
    theta_hat = (thetax_hat - 1)/(thetay_hat - 1) - fit_hat #(tau-1)/(alpha-1) - snz
    
    
    #HOW TO GENERATE JACKKNIFE:
        #Dual nesting over nx and ny, generate theta_jk of length nx*ny.
        #for each i,j in nx,ny respectively (nested for):
        #   (1) calculate x_cur_pl = np.delete(xc,i)
        #   (2) calculate y_cur_pl = np.delete(yc,j)
        #   (3) calculate logx_cur_fit = np.delete(logxc,i)
        #   (4) calculate logy_cur_fit = np.delete(logyc,i)
        #   (5) calculate tau from x_cur_pl, alpha from y_cur_pl, snz from logx_cur_fit and logy_cur_fit
        #   (6) jackknife value at [i,j] is (tau-1)/(alpha-1) - snz
        
    nx = len(xc)
    ny = len(yc)
        
    #jackknife over thetax and fit_x
    thetax_jk = np.zeros(nx)
    thetafit_jk = np.zeros(nx)
    for i in range(nx):
        tmpx = np.delete(xc,i)
        tmplogx = np.delete(logxc,i)
        tmplogy = np.delete(logyc,i)
        thetax_jk[i] = find_pl_exact(tmpx,xmin,xmax)[0]
        thetafit_jk[i] = scipy.stats.linregress(tmplogx,tmplogy).slope
    
    #jackknife over thetay
    thetay_jk = np.zeros(ny)
    for i in range(ny):
        thetay_jk[i] = find_pl_exact(np.delete(yc,i),ymin,ymax)[0]
                
    theta_jk = np.zeros(nx*ny)
    #construct jackknife distribution
    for i in range(nx):
        for j in range(ny):
            theta_jk[int(j*nx + i)] = (thetax_jk[i] - 1)/(thetay_jk[j] - 1) - thetafit_jk[i]
    
    #bootstrap_estimates must be lhss and bootstrap_estimates2 must be snz    
    bootstrap_estimates = bootstrap_estimates - bootstrap_estimates2
    
    return bca_core(theta_hat,theta_jk,bootstrap_estimates,alpha)

#calculate the BCa confidence interval for a given type, given a set of bootstrap estimates from bootstrap().
def bca(x,xmin,xmax, bootstrap_estimates, mytype = 'power_law', y = None, ymin = None, ymax = None, ci = 0.95, bootstrap_estimates2 = None):
    
    if mytype == 'power_law':
        return bca_pl(x,xmin,xmax,bootstrap_estimates, ci = ci)
    #for all options below this, y, ymin, and ymax must be given.
    if y is None:
        print("error. For mytype == fit or mytype == lhs, please give both x and y and their corresponding min, max.")
        return 1,1,1    
    if mytype == 'fit':
        return bca_fit(x,y,xmin,xmax,bootstrap_estimates, ci = ci)
    if mytype == 'lhs':
        return bca_lhs(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates, ci= ci)
    if bootstrap_estimates2 is None:
        print("error. For exponent relationship, please give both bootstrap_estimates (for i.e. (tau-1)/(alpha-1)) and bootstrap_estimates2 (for i.e. snz)")
        return 1,1,1
    if mytype == 'rel':
        return bca_rel(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates,bootstrap_estimates2, ci=ci)

    #if it's gotten to this point, the option is incorrect.
    print('Error. Please input mytype == power_law, lhs, fit, or rel.')
    return 1,1,1
    

#vm = vector of max velocities. The expected scaling relationship is (tau-1)/(mu-1) = sp for vm vs size
#The expected scaling relationship is (alpha-1)/(mu-1) = p/(nz) for vm vs duration

#speed this up
#@nb.njit #jit fails
def bootstrap(s,d,vm,smin,smax,dmin,dmax,num_runs = 10000,is_fixed = False, mytype = 'power_law_exact',dex = 0.25, kic = ' ',min_events = 10,ctr_max = 10):
    cursmin = -1
    cursmax = -1
    curdmin = -1
    curdmax = -1
    ctr = 0
    #ctr_max is the max number of times to try reshuffling before skipping
    #min_events is the minimum number of scaling events required
    taus = np.array([np.nan]*num_runs)
    alphas = np.array([np.nan]*num_runs)
    lhss = np.array([np.nan]*num_runs)

    mus = np.array([np.nan]*num_runs) #exponent on vmax CCDF
    vlhss = np.array([np.nan]*num_runs) #relationship (tau-1)/(mu-1)
    vdlhss = np.array([np.nan]*num_runs) #relationship (alpha-1)/(mu-1)
    
    snzs = np.array([np.nan]*num_runs)
    sps = np.array([np.nan]*num_runs) #vmax vs size power law exponent
    pnzs = np.array([np.nan]*num_runs) #vmax vs duration power law exponent (rho)/(nu*z)

    num_avs = len(s)
    curs = np.zeros(num_avs)
    curd = np.zeros(num_avs)
    curvm = np.zeros(num_avs)
    
    #hold log of s and d so it doesnt have to be recalculated every run
    logs = np.log(s)
    logd = np.log(d)
    logvm = np.log(vm)
    
    #calculate boundaries on vmax for ccdf from vmax vs size
    #old method. Works, but the issue is that since vmin and vmax are determined from smin and smax, we get
    #too much variance when varying the boundaries by +- 0.5 decades. Usually vmax only varies over 1 decade or so total.
    sbin,vbin,_ = logbinning(s,vm,50)
    myinterp = interp1d(np.log10(sbin),np.log10(vbin))
    
    if len(s[(s >= smin)*(s <= smax)]) <= min_events or len(d[(d >= dmin)*(d <= dmax)]) <= min_events:
        print('Not enough events to bootstrap. Returning...')
        return taus,alphas,mus, lhss,vlhss,vdlhss, snzs,sps,pnzs
        
    
    tmpt = -1
    tmpa = -1
    tmpmu = -1
    tmpsp = -1
    if mytype == 'power_law':
        fun = find_pl_fast #set fun to be the power_law() function
    elif mytype == 'power_law_exact':
        fun = find_pl_exact #get the exact solution
    elif mytype == 'truncated_power_law':
        fun = find_tpl #set fun to be the find_tpl() function instead
    else:
        print('Wrong option!')
        return
    
    for i in range(num_runs):
        ctr = 0
        if i//1000 == i/1000:
            print(i)
        #print(i)
        tmp = np.random.random(size = (1,num_avs))[0]
        randints = myround((num_avs-1)*tmp)
        
        #get random subsets of data where s and d are pulled from same list of avalanches
        sc = s[randints]
        dc = d[randints]
        vmc = vm[randints]
        
        #calculate boundaries
        cursmax = smax
        cursmin = smin
        curdmax = dmax
        curdmin = dmin
        curvmin = 10**myinterp(max(np.log10(cursmin),min(np.log10(sbin))))
        curvmax = 10**myinterp(min(np.log10(cursmax),max(np.log10(sbin))))
        if is_fixed == False:
            #new method: +- 0.5 decades in size/duration
            cursmax = smax*10**(random.uniform(-dex,dex))#Old:random.uniform(0.36,1.64)
            cursmin = smin*10**(random.uniform(-dex,dex))
            curdmax = dmax*10**(random.uniform(-dex,dex))
            curdmin = dmin*10**(random.uniform(-dex,dex))
            ""
            curvmin = 10**myinterp(max(np.log10(cursmin),min(np.log10(sbin))))
            curvmax = 10**myinterp(min(np.log10(cursmax),max(np.log10(sbin))))
            
            scc = sc[(sc >= cursmin)*(sc <= cursmax)]
            dcc = dc[(dc >= curdmin)*(dc <= curdmax)] 
            vmcc=vmc[(vmc >= curvmin)*(vmc <= curvmax)]
            
            if len(scc) > 3 and len(dcc) > 3 and len(vmcc) > 3 and not all(scc == scc[0]*np.ones(len(scc))) and not all(dcc == dcc[0]*np.ones(len(dcc))) and not all(vmcc == vmcc[0]*np.ones(len(vmcc))):
            
                tmpt = fun(scc,cursmin,xmax = cursmax)[0]
                tmpa = fun(dcc,curdmin,xmax = curdmax)[0]
                tmpmu = fun(vmcc,curvmin,xmax = curvmax)[0]
            else:
                tmpt = np.nan
                tmpa = np.nan
                tmpmu = np.nan
            while ctr < ctr_max and (cursmin >= cursmax or curdmin >= curdmax or len(scc) <= 3 or len(dcc) <= 3 or len(vmcc) <= 3 or all(scc == scc[0]*np.ones(len(scc))) or all(dcc == dcc[0]*np.ones(len(dcc))) or all(vmcc == vmcc[0]*np.ones(len(vmcc)))):
                cursmax = smax*10**(random.uniform(-dex,dex))#Old:random.uniform(0.36,1.64)
                cursmin = smin*10**(random.uniform(-dex,dex))
                curdmax = dmax*10**(random.uniform(-dex,dex))
                curdmin = dmin*10**(random.uniform(-dex,dex))
                curvmin = 10**myinterp(max(np.log10(cursmin),min(np.log10(sbin))))
                curvmax = 10**myinterp(min(np.log10(cursmax),max(np.log10(sbin))))
                
                scc = sc[(sc >= cursmin)*(sc <= cursmax)]
                dcc = dc[(dc >= curdmin)*(dc <= curdmax)]
                vmcc=vmc[(vmc >= curvmin)*(vmc <= curvmax)]
                if len(scc) == 0:
                    scc = [0]
                if len(dcc) == 0:
                    dcc = [0]
                if len(vmcc) == 0:
                    vmcc = [0]
                try:
                    tmpt = fun(scc,cursmin,xmax = cursmax)[0]
                    tmpa = fun(dcc,curdmin,xmax = curdmax)[0]
                    tmpmu = fun(vmcc,curvmin,xmax = curvmax)[0]
                except:
                    tmpt = np.nan
                    tmpa = np.nan
                    tmpmu = np.nan
                ctr = ctr + 1
        else:
            cursmax = smax
            cursmin = smin
            curdmax = dmax
            curdmin = dmin
            curvmin = 10**myinterp(max(np.log10(cursmin),min(np.log10(sbin))))
            curvmax = 10**myinterp(min(np.log10(cursmax),max(np.log10(sbin))))
        
            scc = sc[(sc >= cursmin)*(sc <= cursmax)]
            dcc = dc[(dc >= curdmin)*(dc <= curdmax)]
            vmcc=vmc[(vmc >= curvmin)*(vmc <= curvmax)]
            
            if len(scc) > 3 and len(dcc) > 3 and len(vmcc) > 3:
            
                tmpt = fun(scc,cursmin,xmax = cursmax)[0]
                tmpa = fun(dcc,curdmin,xmax = curdmax)[0]
                tmpmu = fun(vmcc,curvmin,xmax = curvmax)[0]
            else:
                tmpt = np.nan
                tmpa = np.nan
                tmpmu = np.nan
            
            
        #get the range of s, vmax, and d to fit  
        logsmin = np.log(cursmin)
        logsmax = np.log(cursmax)
        
        logdc = logd[randints]
        logvmc = logvm[randints]
        logsc = logs[randints]
        
        logdc = logdc[(logsc >= logsmin)*(logsc <= logsmax)]
        logvmc = logvmc[(logsc >= logsmin)*(logsc <= logsmax)]        
        logsc = logsc[(logsc >= logsmin)*(logsc <= logsmax)]
        
        
        #calculate power laws
        if ctr < ctr_max:
            taus[i] = tmpt
            alphas[i] = tmpa
            mus[i] = tmpmu
            lhss[i] = (tmpt-1)/(tmpa-1)
            vlhss[i] = (tmpt-1)/(tmpmu-1)
            vdlhss[i] = (tmpa-1)/(tmpmu-1)
        else:
            print('count exceeded kic + ' + kic)
            lhss[i] = np.nan
            vlhss[i] = np.nan
            vdlhss[i] = np.nan
            taus[i] = np.nan
            alphas[i] = np.nan
            mus[i] = np.nan
        
        if ctr < ctr_max and len(logsc) > 3 and len(logdc) > 3 and len(logvmc) > 3:
            snzs[i] = scipy.stats.linregress(logsc,y = logdc).slope
            sps[i] = scipy.stats.linregress(logsc,y = logvmc).slope

            #get the vmax vs duration exponent. Need to re-do above analysis, but with *duration limits*.
            #this ensures the duration limit stuff is independent from the size limit stuff.
            logdmin = np.log(curdmin)
            logdmax = np.log(curdmax)
        
            logvmc = logvm[randints]
            logdc = logd[randints]
            
            logvmc = logvmc[(logdc >= logdmin)*(logdc <= logdmax)]
            logdc = logdc[(logdc >= logdmin)*(logdc <= logdmax)]
            
            pnzs[i] = scipy.stats.linregress(logdc,y = logvmc).slope
        else:
            snzs[i] = np.nan
            sps[i] = np.nan
            pnzs[i] = np.nan
                
        
    return taus,alphas,mus, lhss,vlhss,vdlhss, snzs,sps,pnzs


"""
#Future! Make bootstrap run in parallel to make it faster.
#wrapper function for bootstrap
def boot_star(args):
    return bootstrap(*args)

from multiprocessing import Pool
def boot_multi(s,d,vm,smin,smax,dmin,dmax,num_runs = 10000,is_fixed = False, mytype = 'power_law'):
    #get number of threads
    threads = os.cpu_count()
    args = (s,d,vm,smin,smax,dmin,dmax,num_runs//threads,is_fixed, mytype) #create tuple to hold each branch of bootstrap
    pool = Pool(threads) #initialize pool
    out = pool.map(boot_star,args)
    return out
"""    

#get the confidence intervals from the array of bootstrapped values.
#95% confidence interval is default. That is, 95% of values are going to be in range
#(lo,hi)
def confidence_intervals(vals,ci = 0.95):
    ci = 100 - ci
    mu = np.nanmedian(vals)
    lo = np.nanpercentile(vals,ci/2)
    hi = np.nanpercentile(vals,100-ci/2)
    return mu, lo, hi
"""
get the z-score from the values. Note that the variance of the vector of bootstrapped values
is the standard error of the mean (SEM) of that estimator!!

**That is, you do NOT need to divide by sqrt(N) when calculating the z-score**
Dividing by sqrt(N) leads to the unphysical situation where taking more bootstrapping
samples increases the certainty of your measurement. BOOTSTRAPPING ONLY HELPS YOU ESTIMATE
THE STATISTICS OF YOUR ESTIMATOR!!

p-value is certainty that mua =/= mub. p < 0.01 is significant at the 1% level, p < 0.05 is significant at the 5% level.

A high p-value does not necessarily mean that the two histograms do not overlap with one another!

IT IS PROBABLE THAT THIS FUNCTION IS OUTPERFORMED BY TOST!!
"""
def zscore(valsa,valsb):
    mua,loa,hia = confidence_intervals(valsa,ci = 68) #get the 68% confidence interval
    mub,lob,hib = confidence_intervals(valsb,ci = 68)
    
    
    #assume greatest possible error when estimating a normal distribution from valsa and valsb, which are generally not normal
    siga = max(hia-mua,mua-loa)
    sigb = max(hib-mub,mub-lob)
    
    z = abs(mua-mub)/np.sqrt(siga**2 + sigb**2) #NO divide by N here!!
    p = scipy.stats.norm.sf(z)*2
    return z,p #The means are only significantly different if p <= 0.2 (i.e. if there is a 20% chance the differences in means arose by chance)

"""
Perform a two-sided one-tailed t-test of equivalence (TOST) on the difference of means between the histograms valsa and valsb.

The two-sided one-tailed t-test has two null hypotheses: (mu1-mu2) < lo or (mu1-mu2) > hi for user-defined (lo,hi) = (-sig,sig)
at a given p value threshold, defaults to p <= 0.05.

If (mu1-mu2) < lo AND (mu1-mu2) > hi are rejected, then it must be that lo < (mu1 - m2) < hi at the confidence level supplied.

***LEVEL 2 DETERMINATION OF EXPONENT RELATIONSHIP***
We are not only interested in if lo < (mu1-mu2) < hi, but also if (mu1-mu2) is near enough to zero.
If we show that lo < mu1 - mu2 < hi, and 0 is within the 68% CI of the mean value, then we can come to the conclusion that
(mu1-mu2) is consistent with zero and the difference in means between these quantities is not statistically significant.
***This is the most strict version of checking if our exponent relationship holds!***
***LEVEL 2 END***

***LEVEL 1 DETERMINATION OF EXPONENT RELATIONSHIP***
We have a looser definition, in which we see if *just* the 68% CI of (mu1-mu2) has 0 in its bounds. This would mean that
zero is consistent with (mu1-mu2) without necessarily determining that the difference significantly departs from zero.
***LEVEL 1 END***

***LEVEL 0 DETERMINATION OF EXPONENT RELATIONSHIP***
If the 95% CI of mu1-mu2 does not include 0, then we are confident within a 5% type I error that the exponent relationship
does not hold.
***LEVEL 0 END***
"""

#I would not recommend using this right now -- it appears not to give us answers as significant when we indeed believe them to be!

def tost(valsa,valsb,sig = 0.2):
    conf = 0 #level of confidence of determination.
    dist = valsa-valsb #construct the distribution of differences
    med,lo,hi = confidence_intervals(dist,ci = 68) #get median value med, with 68% CI = (lo,hi)
    lo = med - lo
    hi = hi - med #convert CI to difference from median
    zlo = (abs(med) + sig)/lo
    zhi = (abs(med) - sig)/hi
    """
    #If one wants both p values for low and high
    plo = scipy.stats.norm.sf(zlo)
    phi = 1-scipy.stats.norm.sf(zhi)
    """
    
    z = min(abs(zlo),abs(zhi))
    
    p = scipy.stats.norm.sf(z)

    return z,p

#Input is vectors of size and duration
def ad(s,d,smin,smax,dmin,dmax):
    
    #scaling events
    sc = s[(s >= smin)*(s <= smax)]
    dc = d[(d >= dmin)*(d <= dmax)]
    
    #get events larger than smin to compare for truncated power law
    scm = s[(s >= smin)]
    dcm = d[(d >= dmin)]

    #power law
    tau_pl = find_pl_exact(sc,smin,smax)
    alpha_pl = find_pl_exact(dc,dmin,dmax)
    
    #truncated power law (no xmax!)
    tau_tpl = find_tpl(scm,smin)
    alpha_tpl = find_tpl(dcm,dmin)
    
    #lognormal
    sizelog = find_lognormal(sc,smin,smax)
    durlog = find_lognormal(dc,dmin,dmax)
    
    #exponential
    sizeexp = find_exp(sc,smin)
    durexp = find_exp(dc,dmin)
    
    #simulated datasets
    spl_sim = pl_gen(np.random.rand(1000),smin,smax,tau_pl[0])
    dpl_sim = pl_gen(np.random.rand(1000),dmin,dmax,alpha_pl[0])
    stpl_sim = tpl_gen(np.random.rand(1000),smin,tau_tpl[0],tau_tpl[1])
    dtpl_sim = tpl_gen(np.random.rand(1000),dmin,alpha_pl[0],alpha_pl[1])
    slog_sim = lognormal_gen(np.random.rand(1000),smin,smax,sizelog[0],sizelog[1])
    dlog_sim = lognormal_gen(np.random.rand(1000),dmin,dmax,durlog[0],durlog[1])   
    sizeexp_sim = exp_gen(np.random.rand(1000),smin,sizeexp[0])
    durexp_sim = exp_gen(np.random.rand(1000),dmin,durexp[0])
    
    #Do AD tests
    ad_spl = scipy.stats.anderson_ksamp([sc, spl_sim])
    ad_dpl = scipy.stats.anderson_ksamp([dc, dpl_sim])
    
    ad_stpl = scipy.stats.anderson_ksamp([scm, stpl_sim])
    ad_dtpl = scipy.stats.anderson_ksamp([dcm, dtpl_sim])
    
    ad_slog = scipy.stats.anderson_ksamp([sc,slog_sim])
    ad_dlog = scipy.stats.anderson_ksamp([dc,dlog_sim])
    
    ad_sizeexp = scipy.stats.anderson_ksamp([sc, sizeexp_sim])
    ad_durexp = scipy.stats.anderson_ksamp([dc, durexp_sim])
    
    
    return ad_spl, ad_stpl, ad_slog, ad_sizeexp, ad_dpl, ad_dtpl, ad_dlog, ad_durexp


##DEFUNCT FUNCTIONS

"""
#same as bootstrap above, but with assumption that duration bins inform the size and vmax bins.
#Testing suggests the error is unphysical.
def bootstrap_dur(s,d,vm,dmin,dmax,num_runs = 10000,is_fixed = False, mytype = 'power_law'):
    cursmin = -1
    cursmax = -1
    curdmin = -1
    curdmax = -1
    taus = np.zeros(num_runs)
    alphas = np.zeros(num_runs)
    lhss = np.zeros(num_runs)

    mus = np.zeros(num_runs) #exponent on vmax CCDF
    vlhss = np.zeros(num_runs) #relationship (tau-1)/(mu-1)
    vdlhss = np.zeros(num_runs) #relationship (alpha-1)/(mu-1)
    
    snzs = np.zeros(num_runs)
    sps = np.zeros(num_runs) #vmax vs size power law exponent
    pnzs = np.zeros(num_runs) #vmax vs duration power law exponent (rho)/(nu*z)

    num_avs = len(s)
    curs = np.zeros(num_avs)
    curd = np.zeros(num_avs)
    curvm = np.zeros(num_avs)
    
    #hold log of s and d so it doesnt have to be recalculated every run
    logs = np.log(s)
    logd = np.log(d)
    logvm = np.log(vm)
    
    #calculate boundaries on vmax for ccdf from vmax vs size
    #old method. Works, but the issue is that since vmin and vmax are determined from smin and smax, we get
    #too much variance when varying the boundaries by +- 0.5 decades. Usually vmax only varies over 1 decade or so total.
    
    dbin,sbind,_ = logbinning(d,s,50)
    myinterpd = interp1d(np.log10(dbin),np.log10(sbind))
    
    sbin,vbin,_ = logbinning(s,vm,50)
    myinterp = interp1d(np.log10(sbin),np.log10(vbin))
    
    tmpt = -1
    tmpa = -1
    tmpmu = -1
    tmpsp = -1
    
    for i in range(num_runs):
        print(i)
        tmp = np.random.random(size = (1,num_avs))[0]
        randints = myround((num_avs-1)*tmp)
        
        #get random subsets of data where s and d are pulled from same list of avalanches
        sc = s[randints]
        dc = d[randints]
        vmc = vm[randints]
        
        #calculate boundaries
        curdmax = dmax
        curdmin = dmin
        cursmin = 10**myinterpd(max(np.log10(curdmin),min(np.log10(dbin))))
        cursmax = 10**myinterpd(min(np.log10(curdmax),max(np.log10(dbin))))
        curvmin = 10**myinterp(max(np.log10(cursmin),min(np.log10(sbin))))
        curvmax = 10**myinterp(min(np.log10(cursmax),max(np.log10(sbin))))
        
        if mytype == 'power_law':
            fun = find_pl_fast #set fun to be the power_law() function
        elif mytype == 'truncated_power_law':
            fun = find_tpl_fast #set fun to be the find_tpl() function instead
        else:
            print('Wrong option!')
            return
        
        if is_fixed == False:
            ""
            #old method: +-64%
            cursmax = smax*random.uniform(0.36,1.64)#Old:random.uniform(0.36,1.64)
            cursmin = smin*random.uniform(0.36,1.64)
            curdmax = dmax*random.uniform(0.36,1.64)
            curdmin = dmin*random.uniform(0.36,1.64)
            ""
            #new method: +- 0.5 decades in size/duration
            curdmax = dmax*10**(random.uniform(-0.5,0.5))
            curdmin = dmin*10**(random.uniform(-0.5,0.5))
            ""
            cursmin = 10**myinterpd(max(np.log10(curdmin),min(np.log10(dbin))))
            cursmax = 10**myinterpd(min(np.log10(curdmax),max(np.log10(dbin))))
            curvmin = 10**myinterp(max(np.log10(cursmin),min(np.log10(sbin))))
            curvmax = 10**myinterp(min(np.log10(cursmax),max(np.log10(sbin))))
            
            scc = sc[(sc >= cursmin)*(sc <= cursmax)]
            dcc = dc[(dc >= curdmin)*(dc <= curdmax)]
            vmcc=vmc[(vmc >= curvmin)*(vmc <= curvmax)]
            
            tmpt = fun(scc,cursmin,xmax = cursmax)
            tmpa = fun(dcc,curdmin,xmax = curdmax)
            tmpmu = fun(vmcc,curvmin,xmax = curvmax)
            while cursmin >= cursmax or curdmin >= curdmax or tmpt <= 1 or tmpa <= 1 or tmpmu <= 1 or len(scc) <= 3 or len(dcc) <= 3 or len(vmcc) <= 3:
                curdmax = dmax*10**(random.uniform(-0.5,0.5))
                curdmin = dmin*10**(random.uniform(-0.5,0.5))
                ""
                cursmin = 10**myinterpd(max(np.log10(curdmin),min(np.log10(dbin))))
                cursmax = 10**myinterpd(min(np.log10(curdmax),max(np.log10(dbin))))
                curvmin = 10**myinterp(max(np.log10(cursmin),min(np.log10(sbin))))
                curvmax = 10**myinterp(min(np.log10(cursmax),max(np.log10(sbin))))
                
                scc = sc[(sc >= cursmin)*(sc <= cursmax)]
                dcc = dc[(dc >= curdmin)*(dc <= curdmax)]
                vmcc=vmc[(vmc >= curvmin)*(vmc <= curvmax)]
                
                tmpt = fun(scc,cursmin,xmax = cursmax)
                tmpa = fun(dcc,curdmin,xmax = curdmax)
                tmpmu = fun(vmcc,curvmin,xmax = curvmax)
        else:
            curdmax = dmax
            curdmin = dmin
            cursmin = 10**myinterpd(max(np.log10(curdmin),min(np.log10(dbin))))
            cursmax = 10**myinterpd(min(np.log10(curdmax),max(np.log10(dbin))))
            curvmin = 10**myinterp(max(np.log10(cursmin),min(np.log10(sbin))))
            curvmax = 10**myinterp(min(np.log10(cursmax),max(np.log10(sbin))))
            
            scc = sc[(sc >= cursmin)*(sc <= cursmax)]
            dcc = dc[(dc >= curdmin)*(dc <= curdmax)]
            vmcc=vmc[(vmc >= curvmin)*(vmc <= curvmax)]
            
            tmpt = fun(scc,cursmin,xmax = cursmax)
            tmpa = fun(dcc,curdmin,xmax = curdmax)
            tmpmu = fun(vmcc,curvmin,xmax = curvmax)
            
            
        #get the range of s, vmax, and d to fit  
        logsmin = np.log(cursmin)
        logsmax = np.log(cursmax)
        
        logdc = logd[randints]
        logvmc = logvm[randints]
        logsc = logs[randints]
        
        logdc = logdc[(logsc >= logsmin)*(logsc <= logsmax)]
        logvmc = logvmc[(logsc >= logsmin)*(logsc <= logsmax)]        
        logsc = logsc[(logsc >= logsmin)*(logsc <= logsmax)]
        
        
        #calculate power laws
        lhss[i] = (tmpt-1)/(tmpa-1)
        vlhss[i] = (tmpt-1)/(tmpmu-1)
        vdlhss[i] = (tmpa-1)/(tmpmu-1)
        
        taus[i] = tmpt
        alphas[i] = tmpa
        mus[i] = tmpmu
        
        snzs[i] = scipy.stats.linregress(logsc,y = logdc).slope
        sps[i] = scipy.stats.linregress(logsc,y = logvmc).slope
        
        #get the vmax vs duration exponent. Need to re-do above analysis, but with *duration limits*.
        #this ensures the duration limit stuff is independent from the size limit stuff.
        logdmin = np.log(curdmin)
        logdmax = np.log(curdmax)

        logvmc = logvm[randints]
        logdc = logd[randints]
        
        logvmc = logvmc[(logdc >= logdmin)*(logdc <= logdmax)]
        logdc = logdc[(logdc >= logdmin)*(logdc <= logdmax)]
        
        pnzs[i] = scipy.stats.linregress(logdc,y = logvmc).slope
        
    return taus,alphas,mus, lhss,vlhss,vdlhss, snzs,sps,pnzs




def ad_testing(path,fname,idx,printstuff = False):
    
    kic = fname[2:-4]
    mytime,myflux = load_file(path,othernames[idx])
    v,t,s,d,st,en = gs(velocity = np.array(myflux)/np.median(myflux)-1, time = np.array(mytime),shapes = True,mindrop = 1e-9)
    
    #mle
    tau_mle = find_pl_mo(s,smins[idx],smaxs[idx])
    alpha_mle = find_pl_mo(d,dmins[idx],dmaxs[idx])
    
    sizelog = find_lognormal(s,smins[idx],smaxs[idx])
    durlog = find_lognormal(d,dmins[idx],dmaxs[idx])
    
    #get the simulated datasets to test against
    smle_sim = pl_gen(np.random.rand(1000),smins[idx],smaxs[idx],tau_mle[0])
    dmle_sim = pl_gen(np.random.rand(1000),dmins[idx],dmaxs[idx],alpha_mle[0])
    slog_sim = [lognormal_gen(random.random(),smins[idx],smaxs[idx],sizelog[0],sizelog[1]) for i in range(1000)]
    dlog_sim = [lognormal_gen(random.random(),dmins[idx],dmaxs[idx],durlog[0],durlog[1]) for i in range(1000)]    
    
    
    s_scale = s[(s >= smins[idx]) * (s <= smaxs[idx])]
    d_scale = d[(d >= dmins[idx]) * (d <= dmaxs[idx])]
    
    ad_smle = scipy.stats.anderson_ksamp([s_scale, smle_sim])
    ad_dmle = scipy.stats.anderson_ksamp([d_scale, dmle_sim])
    
    
    ks_smle = scipy.stats.ks_2samp(s_scale,smle_sim)
    ks_dmle = scipy.stats.ks_2samp(d_scale,dmle_sim)

    ad_slog = scipy.stats.anderson_ksamp([s_scale,slog_sim])
    ad_dlog = scipy.stats.anderson_ksamp([d_scale,dlog_sim])
    #AD tests
    if printstuff:
        print('+++++')
        print('Anderson-Darling tests (significance > 0.1 means consistent w/ power law. p = 0.25 is max reported by AD test.)')
        print('Tau MLE A = %.3f, p = %.3f' % (ad_smle.statistic, ad_smle.significance_level))
        print('Alpha MLE A = %.3f, p = %.3f' % (ad_dmle.statistic, ad_dmle.significance_level))
        print('+++++')
        print('Kolmogorov-Smirnov tests (significance > 0.1 is consistent with power law)')
        print('Tau MLE D = %.3f, p = %.3f' % (ks_smle.statistic, ks_smle.pvalue))
        print('Alpha MLE D = %.3f, p = %.3f' % (ks_dmle.statistic, ks_dmle.pvalue))
        print('+++++')
        print('Alternate hypothesis Anderson-Darling test')
        print('Size Lognormal A = %.3f, p = %.3f' % (ad_slog.statistic,ad_slog.significance_level))
        print('Duration Lognormal A = %.3f, p = %.3f' % (ad_dlog.statistic,ad_dlog.significance_level))
        print('+++++')
    
    return


def find_pl_mo(x,xmin,xmax = 1e6):
    mymean = lambda a: -pl_like(x,a,xmin,xmax)[0]
    myfit = optimize.minimize(mymean,2,method = 'Nelder-Mead').x[0]
    
    lu = lambda a: -pl_like(x,a,xmin*0.36,xmax*1.64)[0]
    if xmin*1.64 < xmax*0.36:
        ul = lambda a: -pl_like(x,a,xmin*1.64,xmax*0.36)[0]
    else:
        ul = lu
    uu = lambda a: -pl_like(x,a,xmin*1.64,xmax*1.64)[0]
    ll = lambda a: -pl_like(x,a,xmin*0.36,xmax*0.36)[0]
    lc = lambda a: -pl_like(x,a,xmin*0.36,xmax*1)[0]
    cl = lambda a: -pl_like(x,a,xmin*1,xmax*0.36)[0]
    cu = lambda a: -pl_like(x,a,xmin*1,xmax*1.64)[0]
    uc = lambda a: -pl_like(x,a,xmin*1.64,xmax*1)[0]

    llr = pl_like(x,myfit,xmin,xmax)[0]
    lu = optimize.minimize(lu,2,method = 'Nelder-Mead').x[0]
    ul = optimize.minimize(ul,2,method = 'Nelder-Mead').x[0]
    uu = optimize.minimize(uu,2,method = 'Nelder-Mead').x[0]
    ll = optimize.minimize(ll,2,method = 'Nelder-Mead').x[0]
    lc = optimize.minimize(lc,2,method = 'Nelder-Mead').x[0]
    cl = optimize.minimize(cl,2,method = 'Nelder-Mead').x[0]
    cu = optimize.minimize(cu,2,method = 'Nelder-Mead').x[0]
    uc = optimize.minimize(uc,2,method = 'Nelder-Mead').x[0]
    tmp = [lu, ul, uu, ll, lc, cl, cu, uc]
    #tmp = [i for i in tmp if i > 1]
    lower = myfit - np.nanmin(tmp)
    upper = np.nanmax(tmp) - myfit
    return myfit, lower, upper, llr

#test bootstrapping to find the true errorbars on power law alpha.
def bootstrap_pl(x,xmin,xmax,option = 'rand'):
    myround = np.vectorize(round)
    #get a subsample
    curmin = 0
    curmax = 0
    x = np.array(x)
    alphas = np.zeros(10000)
    for i in range(10000):
        print(i)
        tmp = np.array([random.random() for i in range(len(x))])
        randints = myround(len(x)*tmp -1)
        randset = x[randints]
        
        #ensure legal values of curmax and curmin
        if option == 'rand':
            curmax = xmax*random.uniform(0.36,1.64)#curmax = (0.82*random.random() + 0.82)*xmax
            curmin = xmin*random.uniform(0.36,1.64)
            while curmin >= curmax or alphas[i] <= 1:
                curmax = random.uniform(0.36*xmax,1.64*xmax)#curmax = (0.82*random.random() + 0.82)*xmax
                curmin = random.uniform(0.36*xmin,1.64*xmin)

                alphas[i] = find_pl_fast(randset,curmin,curmax)
        else:
            curmax = xmax
            curmin = xmin
            alphas[i] = find_pl_fast(randset,curmin,curmax)
    return alphas

#testing function for bootstrapping. Defunct.
def test_bootstrap(path,fname,idx,option = 'rand'):
    #test the bootstrapping method on star fname.
    kic = fname[2:-4]
    mytime,myflux = load_file(path,othernames[idx])
    v,t,s,d,st,en = gs(velocity = np.array(myflux)/np.median(myflux)-1, time = np.array(mytime),shapes = True,mindrop = 1e-9)
    
    #mle
    tau_mle = find_pl_mo(s,smins[idx],smaxs[idx])
    alpha_mle = find_pl_mo(d,dmins[idx],dmaxs[idx])
    
    sizelog = find_lognormal(s,smins[idx],smaxs[idx])
    durlog = find_lognormal(d,dmins[idx],dmaxs[idx])
    
    #bootstrap
    tau_boot = bootstrap_pl(s,smins[idx],smaxs[idx],option = option)
    alpha_boot = bootstrap_pl(d,dmins[idx],dmaxs[idx],option = option)
    
    
    #get mean and error bars of bootstrap method (95% CI)
    ci = 5 #(1-CI) in percent
    mean_tau_boot = np.median(tau_boot) #median is better
    up_tau_boot = np.percentile(tau_boot,100-ci/2) - mean_tau_boot
    lo_tau_boot = mean_tau_boot - np.percentile(tau_boot,ci/2)
    mean_alpha_boot = np.median(alpha_boot)
    up_alpha_boot = np.percentile(alpha_boot,100-ci/2) - mean_alpha_boot
    lo_alpha_boot = mean_alpha_boot - np.percentile(alpha_boot,ci/2)
    
    #get the (tau-1)/(alpha-1)
    tmp = (tau_boot - 1)/(alpha_boot - 1)
    mean_lhs = np.median(tmp)
    up_lhs_boot = np.percentile(tmp,100-ci/2) - mean_lhs
    lo_lhs_boot = mean_lhs - np.percentile(tmp,ci/2)
    
    #get the simulated datasets to test against
    smle_sim = pl_gen(np.random.rand(1000),smins[idx],smaxs[idx],tau_mle[0])
    dmle_sim = pl_gen(np.random.rand(1000),dmins[idx],dmaxs[idx],alpha_mle[0])
    sboot_sim = pl_gen(np.random.rand(1000),smins[idx],smaxs[idx],np.median(tau_boot))
    dboot_sim = pl_gen(np.random.rand(1000),smins[idx],smaxs[idx],np.median(alpha_boot))
    
    slog_sim = [lognormal_gen(random.random(),smins[idx],smaxs[idx],sizelog[0],sizelog[1]) for i in range(1000)]
    dlog_sim = [lognormal_gen(random.random(),dmins[idx],dmaxs[idx],durlog[0],durlog[1]) for i in range(1000)]    
    
    #get the data in the scaling regime defined exactly by xmin and xmax
    s_scale = s[(s >= smins[idx]) * (s <= smaxs[idx])]
    d_scale = d[(d >= dmins[idx]) * (d <= dmaxs[idx])]
    
    print('Mean exponent values found:')
    print('Tau MLE = %.3f + %.3f - %.3f' %(tau_mle[0],tau_mle[2],tau_mle[1]))
    print('Tau bootstrap = %.3f + %.3f - %.3f' % (mean_tau_boot, up_tau_boot, lo_tau_boot))
    print('Alpha MLE = %.3f + %.3f - %.3f' %(alpha_mle[0],alpha_mle[2],alpha_mle[1]))
    print('Alpha bootstrap = %.3f + %.3f - %.3f' % (mean_alpha_boot, up_alpha_boot, lo_alpha_boot))
    print('(t-1)/(a-1) bootstrap = %.3f + %.3f - %.3f' %(mean_lhs, up_lhs_boot, lo_lhs_boot))
    print('(t-1)/(a-1) MLE = %.3f' %((tau_mle[0]-1)/(alpha_mle[0]-1)))
    
    
    
    #AD tests
    print('+++++')
    print('Anderson-Darling tests (significance > 0.1 means consistent w/ power law. p = 0.25 is max reported by AD test.)')
    ad_smle = scipy.stats.anderson_ksamp([s_scale, smle_sim])
    ad_dmle = scipy.stats.anderson_ksamp([d_scale, dmle_sim])
    ad_sboot = scipy.stats.anderson_ksamp([s_scale, sboot_sim])
    ad_dboot = scipy.stats.anderson_ksamp([s_scale, dboot_sim])
    print('Tau MLE A = %.3f, p = %.3f' % (ad_smle.statistic, ad_smle.significance_level))
    print('Tau bootstrap A = %.3f, p = %.3f' % (ad_sboot.statistic, ad_sboot.significance_level))
    print('Alpha MLE A = %.3f, p = %.3f' % (ad_dmle.statistic, ad_dmle.significance_level))
    print('Alpha bootstrap A = %.3f, p = %.3f' % (ad_dboot.statistic, ad_dboot.significance_level))
    print('+++++')
    print('Kolmogorov-Smirnov tests (significance > 0.1 is consistent with power law)')
    ks_smle = scipy.stats.ks_2samp(s_scale,smle_sim)
    ks_dmle = scipy.stats.ks_2samp(d_scale,dmle_sim)
    ks_sboot = scipy.stats.ks_2samp(s_scale,sboot_sim)
    ks_dboot = scipy.stats.ks_2samp(s_scale,dboot_sim)
    print('Tau MLE D = %.3f, p = %.3f' % (ks_smle.statistic, ks_smle.pvalue))
    print('Tau bootstrap D = %.3f, p = %.3f' % (ks_sboot.statistic, ks_sboot.pvalue))
    print('Alpha MLE D = %.3f, p = %.3f' % (ks_dmle.statistic, ks_dmle.pvalue))
    print('Alpha bootstrap D = %.3f, p = %.3f' % (ks_dboot.statistic, ks_dboot.pvalue))
    print('+++++')
    print('Alternate hypothesis Anderson-Darling test')
    ad_slog = scipy.stats.anderson_ksamp([s_scale,slog_sim])
    ad_dlog = scipy.stats.anderson_ksamp([d_scale,dlog_sim])
    print('Size Lognormal A = %.3f, p = %.3f' % (ad_slog.statistic,ad_slog.significance_level))
    print('Duration Lognormal A = %.3f, p = %.3f' % (ad_dlog.statistic,ad_dlog.significance_level))
    print('+++++')
    
    
    return tau_mle, alpha_mle, tau_boot, alpha_boot

"""
#defunct, use the AD distance built in to scipy.
#Tweaked slightly from powerlaw() package, uses our power law CDF.
def power_law_ks_distance(data, alpha, xmin, xmax):
    from numpy import arange, sort, mean, argwhere
    x = np.sort(data[(data >= xmin)* (data <= xmax)])
    n = len(x)
    
    #Actual_CDF = arange(n) / float(n) #gives y value of CDF. X value of CDF comes from csx =np.unique([i for i in s])
    _,Actual_CDF = ccdf(list(x))
    Actual_CDF = 1-Actual_CDF #turn the CCDF into a CDF
    
    X = xmax/xmin
    Xdat = x/xmin
    Theoretical_CDF = (1-Xdat**(1-alpha))/(1-X**(1-alpha))
    
    #calculate max difference between theory and actual cdf
    D_plus = max(Theoretical_CDF - Actual_CDF)
    D_minus = max(Actual_CDF - Theoretical_CDF)
    Kappa = 1 + mean(Theoretical_CDF - Actual_CDF)
    D = max(D_plus, D_minus)

    return D
"""
An (arguably) better way to determine if the data indeed follow a power law distribution
is to use the Anderson-Darling K-Sample test (https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test)

The Anderson-Darling test is preferred over the KS test for heavy-tailed distributions such as the power law
because the AD test is more sensitive to tails than the KS test. It appears to be favored
overall by statisticians, but I am unsure why.

The null hypothesis is that two samples a and b are drawn from the same distribution.
If the significance is below 10%, we can be confident at the 10% level that we can reject this
null hypothesis. In other words, these two samples would have been drawn from the same distribution
only 10% of the time. It is canonical to accept that two samples are likely from the same
distribution if significance >= 10%, and the samples are drawn from the same distribution
for significance < 10%.



#bootstrapping for lognormal. Lognormal function is apparently not concave, so bootstrapping fails.
#defunct.
def bootstrap_lognormal(x,xmin,xmax, num_runs = 10000):
    myround = np.vectorize(round)
    curmin = 0
    curmax = 0
    x = np.array(x)
    mus = np.zeros(num_runs)
    sigmas = np.zeros(num_runs)
    for i in range(num_runs):
        print(i)
        tmp = np.random.random(size = (1,len(s)))[0]
        
        #ensure legal values of curmax and curmin
        curmax = random.uniform(0.36*xmax,1.64*xmax)#curmax = (0.82*random.random() + 0.82)*xmax
        curmin = random.uniform(0.36*xmin,1.64*xmin)
        #only accept values of alpha that are greater than 1. Values less than or equal to 1 are illegal.
        while curmin >= curmax:
            curmax = random.uniform(0.36*xmax,1.64*xmax)#curmax = (0.82*random.random() + 0.82)*xmax
            curmin = random.uniform(0.36*xmin,1.64*xmin)
        randints = myround(len(x)*tmp -1)
        randset = x[randints]
        tmp = find_lognormal(randset,curmin,curmax)
        mus[i] = tmp[0]
        sigmas[i] = tmp[1]
    return mus,sigmas

"""