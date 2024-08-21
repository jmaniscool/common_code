# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:01:11 2024

@author: sickl

Bootstrapping functions and BCA CI estimator functions.

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

import numpy as np
import scipy
import numba


from .logbinning import logbinning
from .likelihoods import find_pl, find_tpl, find_pl_discrete

#bootstrap by picking random subset of avalanches, then restricting analysis to
#smin < s < smax and dmin < d < dmax where smin/smax and dmin/dmax are varied
#*independently* of one another.
#Independent variation is considered so any effect of cross correlation introduced
#by deriving smin/smax from dmin/dmax is not considered; i.e. it errs on the side of
#caution by making error bars larger.


#numba polyfit from https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
# Define Functions Using Numba
# Idea here is to solve ax = b, using least squares, where a represents our coefficients e.g. x**2, x, constants
@numba.njit
def _coeff_mat(x, deg):
    #Set up the matrix of coefficients.
    mat_ = np.zeros(shape=(x.shape[0],deg + 1))
    const = np.ones_like(x)
    mat_[:,0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_
    
@numba.njit
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_
 
@numba.njit
def fit_poly(x, y, deg):
    """
    A Numba safe version of a polynomial fitter.
    
    Source
    https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4

    Parameters
    ----------
    x : float array
        The data on the x axis.
    y : float array
        The data on the y axis.
    deg : int
        The degree of the polynomial.

    Returns
    -------
    coeffs : float array
        The coefficients of the best fit from highest order to lowest order.
        I.e. for Ax^2 + Bx + C, coeffs = [A, B, C].
        

    """
    a = np.zeros(shape=(x.shape[0],deg + 1))
    const = np.ones_like(x)
    a[:,0] = const
    a[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            a[:, n] = x**n
            
    p = np.linalg.lstsq(a, y)[0]
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]




#get a new random xmin_star and xmax_star which varies over -dex to dex in log scale and ensures xmin < xmax.
#note that ensuring xmin > xmax makes the "random" value of xmax covariant with xmin.
@numba.njit
def logrand(xmin,xmax,dex_lo, dex_hi):
    """
    Generate two random numbers, xmin* and xmax*, that are within dex orders of
    magnitude of xmin and xmax respectively and are guaranteed to follow xmin* > xmax*.

    Parameters
    ----------
    xmin : float
        xmin* is generated within the range xmin*10**(-dex) and xmin*10**(dex)
    xmax : float
        xmax* is generated within the range xmax*10**(-dex) and xmax*10**(dex)
        If xmin* > xmax*10**(-dex), then xmin* is used as the lower range for
        xmax*
    dex : float
        The number of decades around which to generate xmin* and xmax* from xmin
        and xmax.

    Returns
    -------
    xmin* : float
        The subsample xmin.
    xmax* : float
        The subsample xmax.

    """
    #if dex is zero, xmin and xmax do not change.
    logxmin = np.log10(xmin)
    logxmax = np.log10(xmax)
    
    logxmin_star = logxmin
    logxmax_star = logxmax
    
    if dex_lo > 0:
        logxmin_star = np.random.uniform(-dex_lo + logxmin, dex_lo + logxmin) #get log(xmin) to vary between -dex and +dex
    if dex_hi > 0:
        logxmax_star = np.random.uniform(max([-dex_hi + logxmax, logxmin_star]), dex_hi + logxmax) #log(xmax_star) varies between -dex and dex if logxmin_star < -dex + logsmax_star, otherwise log(xmax_star) varies between logxmin_star and +dex
    
    xmin_star = 10**logxmin_star
    xmax_star = 10**logxmax_star
    return xmin_star, xmax_star


#find ymin and ymax from an interpolated function over logx,logy. Default to 50 log bins.
@numba.njit
def loginterp(x,y,xmin,xmax, bins = 50):
    """
    Find ymin and ymax from linearly interpolated function over log(x) and log(y).

    Parameters
    ----------
    x : float array
        The values on the x axis.
    y : float array
        The values on the y axis.
    xmin : float
        The float value to extrapolate ymin from.
    xmax : float
        The float value to extrapolate ymax from.
    bins : int, optional
        The number of logarithmic bins to place the data into. The default is 50.

    Returns
    -------
    ymin : float
        The extrapolated float value from xmin.
    ymax : float
        The extrapolated float value from xmax.

    """
    bx,by,_ = logbinning(x,y,bins)
    
    logbx = np.log10(bx)
    logby = np.log10(by)
    
    #myinterp = scipy.interpolate.interp1d(logbx,logby)
    #myinterp = np.interp(logbx)
    lo = max([np.log10(xmin),min(logbx)])
    hi = min([np.log10(xmax),max(logbx)])
    
    logymin = np.interp(lo,logbx,logby)
    logymax = np.interp(hi,logbx,logby)
    
    ymin = 10**logymin
    ymax = 10**logymax
    return ymin,ymax
    

#bootstrapping core. From an input list of avalanche s,d,smin,smax,etc, estimate a single run of exponents.
@numba.njit
def bootstrap_core(s,d,smin, smax, dmin, dmax,vm, vmin, vmax,logs,logd,logvm, sdex_lo, sdex_hi, ddex_lo, ddex_hi, ctr_max, stepsize):
    """
    The bootstrapping core function. Given a set of s,d (and optionally vm),
    generate a single bootstrapped sample of exponents.
    
    For more information, see the main bootstrap() function.

    """
    #All of the logic of the bootstrapping is done here. For more complete documentation, see bootstrap().
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
    
    
    #try to get enough events to bootstrap over (at least 10.)
    while (ctr <= ctr_max)*((nums < 10) + (numd < 10) + (numv < 10)):    
        idxs = np.random.randint(0,length,length) #get indexes of avalanches to sample
        smin_star,smax_star = logrand(smin,smax,sdex_lo, sdex_hi) #get random smin and smax, ensuring smax > smin    
        dmin_star,dmax_star = logrand(dmin,dmax,ddex_lo, ddex_hi) #get random dmin and dmax, ensuring dmax > dmin
        
        sc = s[idxs] #get list of sampled avalanche size, duration, velocity and their logs
        dc = d[idxs]
        vmc = vm[idxs]    
        logsc = logs[idxs]
        logdc = logd[idxs]    
        logvmc = logvm[idxs]
        
        #if vmin and vmax are something other than 1, estimate vm statistics.
        if (vmin != 1)*(vmax != 1):
            vmin_star,vmax_star = logrand(vmin,vmax,sdex_lo, sdex_hi) #get random vmin and vmax, ensuring vmax > vmin        
            
        nums = np.sum((sc >= smin_star)*(sc <= smax_star))
        numd = np.sum((dc >= dmin_star)*(dc <= dmax_star))
        numv = np.sum((vmc >= vmin_star)*(vmc <= vmax_star))
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
    tau = find_pl(scc,smin_star,smax_star)[0]
    if stepsize is None:
        alpha = find_pl(dc,dmin_star,dmax_star)[0]
    else:
        alpha = find_pl_discrete(dc, dmin_star,dmax_star,stepsize)[0] #fix bug whereby double rounding occurs if one uses find_pl_discrete on data that has already been truncated.
    
    #find snz and (tau-1)/(alpha-1)
    #snz = scipy.stats.linregress(logscc,logdcc).slope
    snz = fit_poly(logscc,logdcc,1)[0]
    sdlhs = np.nan
    if alpha > 1:
        sdlhs = (tau-1)/(alpha-1)
    
    #if vm is given, also calculate velocity statistics.
    if (vmin != 1)*(vmax != 1):
        mu = find_pl(vmcc,vmin_star,vmax_star)[0]
        #sp = scipy.stats.linregress(logscc,logvmcc).slope
        #pnz = scipy.stats.linregress(logdcc,logvmcc).slope
        sp = fit_poly(logscc,logvmcc,1)[0]
        pnz = fit_poly(logdcc,logvmcc,1)[0]
        svlhs = (tau-1)/(mu-1)
        dvlhs = (alpha-1)/(mu-1)
    
    
    return tau,alpha,mu, sdlhs,svlhs,dvlhs, snz,sp,pnz

@numba.njit(parallel = True)
def bootstrap_parallel(num_runs,s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,sdex_lo, sdex_hi, ddex_lo, ddex_hi ,ctr_max,stepsize):
    """
    Helper function for bootstrap which parallelizes the task over all available cores.
    
    Each core will run independent bootstrap_core() functions, then at the end combine them into arrays of each exponent's bootstrapped estimates.
    """
    vals = np.zeros((num_runs,9))
    for i in numba.prange(num_runs):
        tmp = bootstrap_core(s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,sdex_lo, sdex_hi, ddex_lo, ddex_hi,ctr_max,stepsize)
        vals[i] = tmp
    
    return vals.transpose()

#v2 of bootstrap, updated Feb 27, 2024. Written to take advantage of various programming fundamentals improvements Jordan learned since the original bootstrap was written
def bootstrap(s,d, smin, smax, dmin, dmax, vm = None, num_runs = 10000, sdex_lo = 0.25, sdex_hi = 0.25, ddex_lo = 0.25, ddex_hi = 0.25, ctr_max = 10, min_events = 10, stepsize = None):
    """
    Generate distributions of exponents that can be used to estimate the confidence interval.
    For each avalanche, the size s, durations d, (and optionally max velocity vm) are inputs.
    Also input the bounds of the size and duration scaling regimes (smin/smax and dmin/dmax respectively).
    Ensure that index i of each of s,d, and vm corresponds to the same (i.e. the ith) avalanche.
    
    Over num_runs runs, select random choices of smin* and smax* (and dmin* and dmax*) which
    are within dex decades of the input smin/smax and dmin/dmax. Then, select a random
    subsample of avalanches with replacement from the s,d vectors. Then, fit tau, alpha, and
    snz from the subsample scaling s and scaling d (i.e. those within the smin*/smax* and
    dmin*/dmax*). Optionally, the scaling regime in vm is interpolated from the regime in s.
    
    Returns 9 arrays of num_runs floats, one for each of the estimated exponents + relations.
    Each array of floats approximates the histogram underlying the distribution of each exponent.

    Parameters
    ----------
    s : float array
        The array of avalanche sizes. Index i corresponds to the i-th avalanche.
    d : float array
        The array of avalanche durations. Index i corresponds to the i-th avalanche, i.e. s[4] and d[4] are the size and duration of the 5th avalanche.
    smin : float
        The minimum size in the scaling regime
    smax : float
        The maximum size in the scaling regime.
    dmin : float
        The minimum duration in the scaling regime.
    dmax : float
        The maximum duration in the scaling regime.
    vm : float array, optional
        If not None, then vm is the array of the avalanche max velocities. The default is None.
    num_runs : int, optional
        The number of runs to use to estimate the histogram. The default is 10000.
    sdex_lo : float, optional
        The number of decades around which smin* can be chosen from smin.
        That is, smin* can be no smaller than 10**-dex_lo times smaller than smin. The default is 0.25.
    sdex_hi : float, optional
        The number of decades around which dmax* can be chosen from around dmax.
        The default is 0.25.        
    ddex_lo : float, optional
        The number of decades around which dmin* can be chosen from dmin. The default is 0.25.
    ddex_hi : float, optional
        The number of decades around which dmax* can be chosen from around dmax. The default is 0.25.
    ctr_max : int, optional
        The number of attempts to make per run to find an appropriate regime. This should not be set much higher than 10. The default is 10.
    min_events : int, optional
        The minimum number of events an run must have to be counted in the bootstrapping. While this value can be set to be as little as 2,
        'reasonable' estimates of the underlying histograms are found only when there is greater than 10 data. The default is 10.
    stepsize : float, optional
        If None, then assume durations are not discretized.
        Otherwise, d is discrete and is parameterized in terms of stepsize, the timestep size.
        That is, all of d should be (close to) multiples of stepsize.
        Durations can be approximated as continuous if dmin > 30*(stepsize) or so. For most cases,
        d_discrete should be True. The default is True.

    Returns
    -------
    taus : float array
        The estimated histogram of size exponent, tau.
    alphas : float array
        The estimated histogram of duration exponent, alpha.
    mus : float array
        If vm is given, then mus is the estimated histogram of the max velocity exponent, mu.
    sdlhss : float array
        The value of (tau-1)/(alpha-1) for each of the runs.
    svlhss : float array
        If vm is given, then svlhss is the value of (tau-1)/(mu-1) for each of the runs.
    dvlhss : float array
        If vm is given, then svlhss is the value of (alpha-1)/(mu-1) for each of the runs.
    snzs : float array
        The estimated histogram of the size versus duration exponent, snz. The expected exponent relationship is (tau-1)/(alpha-1) = snz.
    sps : float array
        If vm is given, then sps is the value of the size versus max velocity expoent, sp. The expected exponent relationship is (tau-1)/(mu-1) = sp
    pnzs : float array
        If vm is given, then pnzs is the value of the duration versus max velocity expoent, p/nz. The expected exponent relationship is (alpha-1)/(mu-1) = p/nz.

    """
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
    else:
        vmin,vmax = loginterp(s,vm,smin,smax, bins = 50) #use for independent vm dex selection, which leads to unreasonably large error bars since vm should have such a small scaling regime most of the time.
        
        bs,bv,_ = logbinning(s,vm,50)
        

    logvm = np.log10(vm)
    
    
    #if there are fewer than min_events number of events in s or d, return.
    if np.sum((s >= smin)*(s <= smax)) <= min_events or np.sum((d >= dmin)*(d <= dmax)) <= min_events:
        print("Not enough events. Returning.")
        return taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs
    
    vals = bootstrap_parallel(num_runs,s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,sdex_lo, sdex_hi, ddex_lo, ddex_hi,ctr_max,stepsize)
    taus = vals[0,:]
    alphas = vals[1,:]
    mus = vals[2,:]
    sdlhss = vals[3,:]
    svlhss = vals[4,:]
    dvlhss = vals[5,:]
    snzs = vals[6,:]
    sps = vals[7,:]
    pnzs = vals[8,:]
        
    return taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs

#the core of the BCa correction given theta_hat and jackknifed samples theta_jk. Validated against scipy.stats.bootstrap.
def bca_core(theta_hat,theta_jk, bootstrap_estimates, alpha):
    """
    Run the core of the BCA algorithm. These are the steps shared between all BCA estimations.
    Specifically, for a bunch of bootstrap estimates bootstrap_estimates and jackknifed resamples theta_jk,
    the BCA algorithm determines how to correct for bias in the boostrap to give asymmetric confidence intervals.
    
    Sources
    Efron B. An Introduction to the Bootstrap (1993)

    Parameters
    ----------
    theta_hat : float
        The optimal value of the parameter, i.e. tau when you do find_pl for smin <= s <= smax.
        If the optimal theta_hat is far from the median of bootstrap_estimates, then the bootstrap estimates are biased.
    theta_jk : float array
        The jackknifed estimates of theta. Obtained by removing one index at a time from the data, then estimating theta.
        The jackknifed estimates are used to estimate the acceleration, i.e. related to the skew
    bootstrap_estimates : float array
        The bootstrap estimates from the bootstrap code.
    alpha : float 
        The confidence to obtain the values at. I.e. alpha = 0.05 for 95% CI.

    Returns
    -------
    theta_hat : float
        The optimal theta.
    ci_lower : float
        The lower bound of the confidence interval, corrected for bias and acceleration.
    ci_upper : float
        The upper bound of the confidence interval, corected for bias and acceleration.

    """
    
    z0 = scipy.stats.norm.ppf((np.sum(bootstrap_estimates < theta_hat) + 0.5) / (len(bootstrap_estimates) + 1))

    theta_jk_dot = np.nanmean(theta_jk)
    
    #SciPy implementation of BCa.
    #NOTE: the numdat term drops out algebraically, so I am unsure why it's there in the SciPy implementation (shown below)
    #top = np.sum((theta_jk_dot - theta_jk)**3/numdat**3)
    #bot = np.sum((theta_jk_dot - theta_jk)**2/numdat**2)
    #a_hat = top/(6*bot**(3/2))
    

    #Equation 14.15 on pp. 186 of "An Introduction to the Bootstrap" by Efron and Tibshirani implementation
    #validated against test data given in the book. For test data a = [48, 36, 20, 29, 42, 42, 20, 42, 22, 41, 45, 14,6, 0, 33, 28, 34, 4, 32, 24, 47, 41, 24, 26, 30,41]
    #given on pp. 180, a_hat should equal 0.61 (given in pp. 186). Formula below gives correct answer.
    #Also checked bca_core against SciPy.
    top = np.nansum((theta_jk_dot - theta_jk)**3)
    bot = np.nansum((theta_jk_dot - theta_jk)**2)
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
    #print(alpha_1)
    ci_lower = np.nanpercentile(bootstrap_estimates,alpha_1*100)
    ci_upper = np.nanpercentile(bootstrap_estimates,alpha_2*100)
    return theta_hat, ci_lower, ci_upper
    
#Find the bias corrected and accelerated (BCa) confidence intervals for power law exponents from the bootstrapping function.
#Returns the true value of the statistic and its lower and upper confidence intervals given output from bootstrap function.
#Method validated against scipy.stats.bootstrap with BCa selected.
def bca_pl(x, xmin,xmax, bootstrap_estimates, ci = 0.95, stepsize = None):
    """
    Obtain the jackknife samples from a power law fit required to pass into
    BCA core. For more details, see the bca_core function.

    Parameters
    ----------
    x : float array
        The data over which to obtain the jackknife samples.
    xmin : float
        The minimum of the scaling regime.
    xmax : float
        The maximum of the scaling regime.
    bootstrap_estimates : float array
        The distribution of bootstrap estimates for the given power law
        from the bootstrap() function.
    ci : float, optional
        The confidence interval to return the BCA confidence intervals in. The default is 0.95.
    stepsize : float, optional
        If not None, then describes the step size between discrete data. The default is None.

    Returns
    -------
    theta_hat : float
        The mean value of the exponent.
    theta_lo : float
        The value of theta at the lower end of the BCA confidence interval.
    theta_hi : float
        The value of theta at the upper end of the BCA confidence interval.

    """
    alpha = 1-ci #get alpha (e.g. confidence interval of 95% means alpha = 0.05)
    
    #edge case: if the bootstrap estimates are all nan, return zeros
    if np.all(np.isnan(bootstrap_estimates)):
        return 0,0,0
    
    #compute the "true" value of theta
    xc = x[(x >= xmin)*(x <= xmax)]
    numdat = len(xc)
    theta_hat = -1
    theta_jk = np.zeros(numdat)
    f = find_pl
    if stepsize is not None:
        #NOTE: when calculating the true value of theta, when the variable is discrete, it must be discretized before inputting into find_pl_discrete.
        x = np.rint(x/stepsize)
        xmin = np.rint(xmin/stepsize)
        xmax = np.rint(xmax/stepsize)
        xc = x[(x >= xmin)*(x <= xmax)]
        f = lambda x,xmin,xmax : find_pl_discrete(x,xmin,xmax,1) #normalize the data before calculating theta_hat

    theta_hat = f(xc,xmin,xmax)[0]
    
    
    #Jackknife estimation of acceleration (from Scipy)
    theta_jk = np.zeros(numdat)
    for i in range(numdat):
        theta_jk[i] = f(np.delete(xc,i),xmin,xmax)[0]
            
    
    if (theta_hat == 1) or (np.sum(theta_jk) == numdat):
        return theta_hat, np.nan, np.nan
    
    #do the bias correction and acceleration
    return bca_core(theta_hat,theta_jk, bootstrap_estimates, alpha)

#compute the BCA confidence intervals for combined statistics. i.e. like (tau-1)/(alpha-1)
def bca_lhs(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates, ci = 0.95, ystepsize = None):
    """
    Obtain the jackknife samples from the left-hand side of the exponent relationship
    (i.e. (tau-1)/(alpha-1) for size and duration) required to pass into BCA core.
    For more details, see the bca_core function.

    Parameters
    ----------
    x : float array
        The data over which the bootstrap samples from the top of the exponent ratio are obtained.
        I.e. for (tau-1)/(alpha-1), x = sizes.
    y : float array
        The data over which the bootstrap samples from the bottom of the exponent ratio are obtained.
        I.e. for (tau-1)/(alpha-1), y = durations.
    xmin : float
        The minimum of the x scaling regime.
    xmax : float
        The maximum of the x scaling regime.
    ymin : float
        The minimum of the y scaling regime.
    ymax : float
        The maximum of the y scaling regime.
    bootstrap_estimates : float array
        The bootstrap estimates of the exponent ratio, i.e. (tau-1)/(alpha-1),
        generated from bootstrap().
    ci : float, optional
        The confidence interval over which to obtain the BCA confidence interval. The default is 0.95.
    stepsize : float, optional
        If not None, then describes the step size between discrete data in the y variable. The default is None.

    Returns
    -------
    theta_hat : float
        The mean value of the exponent.
    theta_lo : float
        The value of theta at the lower end of the BCA confidence interval.
    theta_hi : float
        The value of theta at the upper end of the BCA confidence interval.

    """
    alpha = 1-ci
    
    xc = x[(x >= xmin)*(x <= xmax)]
    yc = y[(y >= ymin)*(y <= ymax)]
    
    #edge case: if the bootstrap estimates are all nan, return zeros
    if np.all(np.isnan(bootstrap_estimates)):
        return 0,0,0
    
    fx = find_pl
    fy = find_pl
    if ystepsize is not None:
        yc = np.rint(y/ystepsize)
        ymin = np.rint(ymin/ystepsize)
        ymax = np.rint(ymax/ystepsize)
        fy = lambda x,xmin,xmax : find_pl_discrete(x,xmin,xmax,1)
    
    thetax_hat = fx(xc,xmin,xmax)[0]
    thetay_hat = fy(yc,ymin,ymax)[0]

    #get the mean value of (tau-1)/(alpha-1)
    theta_hat = (thetax_hat - 1)/(thetay_hat - 1)
    
    nx = len(xc)
    ny = len(yc)
    
    #The real way would be to jackknife over avalanches.
    #That is, remove avalanches (1) one by one (remove same index i,j), (2) compute xc and yc from xmin, xmax, ymin, ymax, (3) compute tau, alpha from the subsample. However, because there are different number of events in the size scaling regime versus the duration scaling regime, this is untenable.
    
    #assuming samples of x and y are independent (they're not! But the resampling in bootstrapping assumes s,d,etc are independent)
    #jackknife over thetax
    thetax_jk = np.zeros(nx)
    for i in range(nx):
        thetax_jk[i] = fx(np.delete(xc,i),xmin,xmax)[0]
    
    #jackknife over thetay
    thetay_jk = np.zeros(ny)
    for i in range(ny):
        thetay_jk[i] = fy(np.delete(yc,i),ymin,ymax)[0]   
    
    #get the total jackknife as an nx*ny array (i.e. removing each of xi and yi then kron-ing them together and flattening)
    theta_jk = np.zeros(nx*ny)
    for i in range(nx):
        for j in range(ny):
            theta_jk[int(j*nx + i)] = (thetax_jk[i] - 1)/(thetay_jk[j] - 1) #compute (tau-1)/(alpha-1) for all jackknifed nx and ny, taking advantage of independence of x and y
        

    if (theta_hat == 1) or (np.sum(theta_jk) == nx*ny):
        return theta_hat, np.nan, np.nan

    return bca_core(theta_hat,theta_jk, bootstrap_estimates, alpha)

#Do BCa on fitted functions (i.e. like snz)
def bca_fit(x,y,xmin,xmax,bootstrap_estimates,ci = 0.95):
    """
    Obtain the jackknife samples from the right-hand side of the exponent relationship
    (i.e. snz for size and duration) required to pass into BCA core.
    For more details, see the bca_core function.

    Parameters
    ----------
    x : float array
        The data on the x-axis of the fitted function relationship.
        I.e. for snz, x = sizes.
    y : float array
        The data on the y-axis of the fitted function relationship.
        I.e. for (tau-1)/(alpha-1), y = durations.
    xmin : float
        The minimum of the x scaling regime.
    xmax : float
        The maximum of the x scaling regime.
    bootstrap_estimates : float array
        The bootstrap estimates of the fited exponent, i.e. snz,
        generated from bootstrap().
    ci : float, optional
        The confidence interval over which to obtain the BCA confidence interval. The default is 0.95.

    Returns
    -------
    theta_hat : float
        The mean value of the exponent.
    theta_lo : float
        The value of theta at the lower end of the BCA confidence interval.
    theta_hi : float
        The value of theta at the upper end of the BCA confidence interval.

    """
    alpha = 1-ci
    
    
    xc = x[(x>= xmin)*(x <= xmax)]
    yc = y[(x >= xmin)*(x <= xmax)]
    
    logxc = np.log10(xc)
    logyc = np.log10(yc)
    
    #edge case: if the bootstrap estimates are all nan, return zeros
    if np.all(np.isnan(bootstrap_estimates)):
        return 0,0,0

    theta_hat = fit_poly(logxc,logyc,1)[0] #replace with jit-able version in case I need the speedup later.
    
    nx = len(xc)
    theta_jk = np.zeros(nx)
    for i in range(nx):
        theta_jk[i] = fit_poly(np.delete(logxc,i),np.delete(logyc,i),1)[0]


    if (theta_hat == 1) or (np.sum(theta_jk) == nx):
        return theta_hat, np.nan, np.nan
        
    return bca_core(theta_hat,theta_jk,bootstrap_estimates,alpha)

#BCa to determine the exponent relationship confidence intervals, i.e. (tau-1)/(alpha-1) = snz
#HOW TO USE: pass in bca_rel(s,d,smin,smax,dmin,dmax,lhss,snzs, ci = 0.95) to get bootstrapping.
def bca_rel(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates,bootstrap_estimates2, ci = 0.95, ystepsize = None):
    """
    Obtain the jackknife samples required for the BCA confidence intervals on an exponent relationship,
    i.e. (tau-1)/(alpha-1) - snz. For more details on the BCA algorithm, see bca_core.

    Parameters
    ----------
    x : float array
        The data on the x-axis of the fitted variable and on the top of the exponent ratio.
        E.g. for (tau-1)/(alpha-1) - snz, x = (sizes)
    y : float array
        The data on the y-axis of the fitted variable and on the bottom of the exponent ratio.
        E.g. for (tau-1)/(alpha-1) - snz, y = (durations)
    xmin : float
        The minimum of the x scaling regime.
    xmax : float
        The maximum of the x scaling regime.
    ymin : float
        The minimum of the y scaling regime.
    ymax : float
        The maximum of the y scaling regime.
    bootstrap_estimates : float array
        The bootstrap estimates of the exponent ratio, i.e. (tau-1)/(alpha-1),
        generated from bootstrap().
    bootstrap_estimates2 : float array
        The bootstrap estimates of the fitted exponent values, i.e. snz,
        generated from bootstrap().
    ci : float, optional
        The confidence interval over which to obtain the BCA confidence interval. The default is 0.95.
    ystepsize : float, optional
        If not None, then describes the step size between discrete data in the y variable. The default is None.

    Returns
    -------
    theta_hat : float
        The mean value of the exponent.
    theta_lo : float
        The value of theta at the lower end of the BCA confidence interval.
    theta_hi : float
        The value of theta at the upper end of the BCA confidence interval.

    """
    alpha = 1 - ci
    
    xc = x[(x>= xmin)*(x <= xmax)]
    yc = y[(y >= ymin)*(y <= ymax)]
    
    normy = y
    normymin = ymin
    normymax = ymax
    
    fx = find_pl
    fy = find_pl
    
    if ystepsize is not None:
        normy = np.rint(y/ystepsize)
        normymin = np.rint(ymin/ystepsize)
        normymax = np.rint(ymax/ystepsize)
        fy = lambda x,xmin,xmax : find_pl_discrete(x,xmin,xmax,1)
        thetay_hat = fy(normy,normymin,normymax)
        
    #edge case: if the bootstrap estimates are all nan, return zeros
    if np.all(np.isnan(bootstrap_estimates)) + np.all(np.isnan(bootstrap_estimates2)):
        return 0,0,0
        
    logxc = np.log10(xc)
    logyc = np.log10(y[(x >= xmin)*(x <= xmax)])

    fit_hat = fit_poly(logxc,logyc,1)[0]
    thetax_hat = fx(xc,xmin,xmax)[0]
    thetay_hat = fy(normy,normymin,normymax)[0] #calculates find_pl on continuous data if ystepsize is not None, otherwise calculates find_pl_discrete on discretized data.
    if thetay_hat == 1:
        thetay_hat = np.nan
    
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
        thetax_jk[i] = fx(tmpx,xmin,xmax)[0]
        thetafit_jk[i] = fit_poly(tmplogx,tmplogy,1)[0]
    
    #jackknife over thetay
    thetay_jk = np.zeros(ny)
    for i in range(ny):
        thetay_jk[i] = fy(np.delete(normy,i),normymin,normymax)[0]
        if thetay_jk[i] == 1:
            thetay_jk[i] = np.nan
                
    theta_jk = np.zeros(nx*ny)
    #construct jackknife distribution
    for i in range(nx):
        for j in range(ny):
            theta_jk[int(j*nx + i)] = (thetax_jk[i] - 1)/(thetay_jk[j] - 1) - thetafit_jk[i]
    
    #bootstrap_estimates must be lhss and bootstrap_estimates2 must be snz
    bootstrap_estimates = bootstrap_estimates - bootstrap_estimates2
    
    if (np.isnan(thetay_hat)) or (np.sum(theta_jk) == nx*ny):
        return theta_hat, np.nan, np.nan
    
    return bca_core(theta_hat,theta_jk,bootstrap_estimates,alpha)

#calculate the BCa confidence interval for a given type, given a set of bootstrap estimates from bootstrap().
def bca(x,xmin,xmax, bootstrap_estimates, mytype = 'power_law', y = None, ymin = None, ymax = None, ci = 0.95, bootstrap_estimates2 = None, stepsize = None):
    """
    Generate the BCA confidence intervals given some bootstrap estimates.
    That is, given the data used to generate the bootstrap estimates, this
    function corrects for bias (median of bootstrap_estimates different from
    from mean value) and acceleration (skewness). See bca_core for more
    details on the BCA algorithm.
    
    Sources
    [1] Efron B. An Introduction to the Bootstrap (1993)

    Parameters
    ----------
    x : float array
        The data over which the bootstrap_estimates were generated.
        For example, x is the array of sizes if bootstrap_estimates are the
        histogram of taus from bootstrap().
    xmin : float
        The minimum of the scaling regmie of x.
    xmax : float
        The maximum of the scaling regime of x.
    bootstrap_estimates : float array
        The bootstrap estimates for the exponent obtained from bootstrap().
    mytype : string, optional
        Tells the BCA algorithm which which kind of BCA estimate to do.
        
        If 'power_law', then assumes that x (with scaling regime xmin <=x<=xmax)
        is generated from find_pl or find_pl_discrete. Use this for tau, alpha, mu.
        
        If 'fit', then attempts to BCA the estimate of exponents obtained from
        fits, i.e. snz for size versus duration, sp for size vs vmax, and p/nz
        for duration vs vmax. Requires y (with scaling regime ymin/ymax) to be
        inputted as the vector of data plotted against the y-axis when computing
        the fit of the exponent. For example, y = durations and x = sizes when
        calculating snz.
        
        If 'lhs', then uses BCA to find the estimate of the left-hand side of
        exponent relationships, i.e. (tau-1)/(alpha-1). Requires x to be the
        vector of data required to generate the top exponent and y to be the
        vector of data required to generate the bottom exponent, i.e. x = sizes
        and y = durations for (tau-1)/(alpha-1). Also requires
        bootstrap_estimates to be set to the bootstrap estimates for the
        exponent relationship from bootstrapping, i.e. (tau-1)/(alpha-1) output
        from bootstrap().
        
        If 'rel', then estimates the confidence interval for a full exponent
        relationship, i.e. (tau-1)/(alpha-1) - snz. Requires x to be set to the
        vector of data used to estimate the top exponent in the exponent ratio.
        Requires y to be set to the vector of data ued to estimate the bottom
        exponent in the exponent ratio. Requires bootstrap_estimates to be
        the left hand side from bootstrap(), i.e. (tau-1)/(alpha-1). Requires
        bootstrap_estimates2 to be set to the  right hand side of the exponent
        relationship, i.e. snz, obtained from bootstrap().
    y : float array, opftional
        If mytype == 'fit', then y is the data vector that represents the data
        plotted on the y-axis when obtaining the fitted exponent, 
        i.e. y = duration for snz.
        
        If mytype == 'lhs' or 'rel', then y is the data vector for the data on the bottom
        of the left-hand side of the exponent relationship, i.e. for (tau-1)/(alpha-1),
        y = duration.
        
        The default is None.
    ymin : float, optional
        The minimum of the scaling regime in y. The default is None.
    ymax : float, optional
        The maximum of the scaling regime in y. The default is None.
    ci : float, optional
        The confidence interval to report the BCA errorbars to (from 0 to 1).
        The default is 0.95. (i.e. a 95% CI)
    bootstrap_estimates2 : float array, optional
        If mytype == 'rel', then this is the float array for the exponent on the 
        right-hand side of the exponent relationship. For example, for
        (tau-1)/(alpha-1) - snz, bootstrap_estimates2 = snzs, the bootstrap outputs
        from bootstrap(). The default is None.
    stepsize : float, optional
        The size of the discrete variable step size. If None, then the variable is
        assumed continuous. The default is None.
        
        Ex) Set stepsize = (timestep size) when fitting for alpha (when my_type == 'power_law' and x = durations)

    Returns
    -------
    theta_hat : float
        The mean value of the exponent.
    theta_lo : float
        The value of theta at the lower end of the BCA confidence interval.
    theta_hi : float
        The value of theta at the upper end of the BCA confidence interval.

    """
    if mytype == 'power_law':
        return bca_pl(x,xmin,xmax,bootstrap_estimates, ci = ci, stepsize = stepsize)
    #for all options below this, y, ymin, and ymax must be given.
    if y is None:
        print("error. For mytype == fit or mytype == lhs, please give both x and y and their corresponding min, max.")
        return 1,1,1    
    if mytype == 'fit':
        return bca_fit(x,y,xmin,xmax,bootstrap_estimates, ci = ci)
    if mytype == 'lhs':
        return bca_lhs(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates, ci= ci, ystepsize = stepsize)
    if bootstrap_estimates2 is None:
        print("error. For exponent relationship, please give both bootstrap_estimates (for i.e. (tau-1)/(alpha-1)) and bootstrap_estimates2 (for i.e. snz)")
        return 1,1,1
    if mytype == 'rel':
        return bca_rel(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates,bootstrap_estimates2, ci=ci, ystepsize = stepsize)

    #if it's gotten to this point, the option is incorrect.
    print('Error. Please input mytype == power_law, lhs, fit, or rel.')
    return 1,1,1

#wrapper that returns the mean value + (1-alpha) confidence intervals for tau, alpha, (tau-1)/(alpha-1), snz, and (tau-1)/(alpha-1) - snz.
def bootstrap_bca(s,d, smin, smax, dmin, dmax, num_runs = 10000, dex = 0.25, ctr_max = 10, min_events = 10, stepsize = None, bca_ci = 0.95, min_alpha = 1.00):
    """
    Wrapper that returns the mean value + (default 95% CI) bias corrected and accelerated
    values of tau, alpha, snz, (tau-1)/(alpha-1), and (tau-1)/(alpha-1) - snz. By default,
    considers only bootstrapping runs where alpha >= min_alpha.

    Parameters
    ----------
    s : float array
        The array of sizes
    d : float array
        The array of durations. If stepsize = None, assume the durations are continuous.
    smin : float
        The minimum of the scaling regime in size.
    smax : float
        The maximum of the scaling regime in size.
    dmin : float
        The minimum of the duration scaling regime.
    dmax : float
        The maximum of the duration scaling regime.
    num_runs : int, optional
        The number of bootstrapping runs to use to obtain the confidence intervals.
        After some testing, it was found that 10k runs is enough to obtain 95% CI
        up to the second decimal point of accuracy. The default is 10000.
    dex : float, optional
        The number of decades of scaling to randomly vary smin/smax and dmin/dmax over
        when bootstrapping. The default is 0.25.
    ctr_max : TYPE, optional
        DESCRIPTION. The default is 10.
    min_events : TYPE, optional
        DESCRIPTION. The default is 10.
    stepsize : TYPE, optional
        DESCRIPTION. The default is None.
    bca_ci : TYPE, optional
        DESCRIPTION. The default is 0.95.
    min_alpha : float, optional
        The minimum value of alpha for 'valid' bootstrappiung runs. In most cases,
        set this to 1, unless the bootstrap returns many alpha = 1 estimates which
        skews the statistics. If you commonly get large values (i.e. >10) in the exponent 
        relationship, set this to 1.01 (some small amount above 1). The default is 1.00.

    Returns
    -------
    taus : float array
        The mean value of tau, followed by its values at the low and high ends of the confidence intervals.
        [tau, tau_lo, tau_hi]
    alphas : float array
        The mean value of alpha, followed by its values at the low and high ends of the confidence intervals.
        If alpha_lo < 1.1 or so, then there is a high chance of divergence in lhss or rels.
        [alpha, alpha_lo, alpha_hi]
    snzs : float array
        The mean value of snz, followed by its values at the low and high ends of the confidence intervals.
        [alpha, alpha_lo, alpha_hi]
    lhss : float array
        The mean value of (tau-1)/(alpha-1), followed by its values at the low and high ends of the confidence intervals.
        If either lhs_lo or lhs_hi are unphysically high (i.e. greater than 10), then set min_alpha to be higher than 1.
        For most cases, set min_alpha = 1.01 to remove from consideration the extremely unlikely runs which have alpha
        close to 1 (causing an asymptote in (tau-1)/(alpha-1)).
        [lhs, lhs_lo, lhs_hi]
    rels : float array
        The mean value of (tau-1)/(alpha-1) - snz, followed by its values at the low and high ends of the confidence intervals.
        If either rel_lo or rel_hi are way too large, set min_alpha to be slightly greater than 1 (i.e. around 1.01) and try
        again. If the problem persists, examine the shape of the histogram by running cc.bootstrap() and plotting the histogram.

    """
    
    taus = np.zeros(3)
    alphas = np.zeros(3)
    lhss = np.zeros(3)
    snzs = np.zeros(3)
    rels = np.zeros(3)
    
    tmp = bootstrap(s,d, smin, smax, dmin, dmax, vm = None, num_runs = num_runs, dex = dex, ctr_max = ctr_max, min_events = min_events, stepsize = stepsize)
    taus_boot = tmp[0]
    alphas_boot = tmp[1]
    lhss_boot = tmp[3]
    snzs_boot = tmp[6]
    
    #get [tau, tau_lo, and tau_hi]
    taus = bca_pl(s,smin,smax, taus_boot[alphas_boot >= min_alpha], ci = bca_ci, stepsize = None)
    
    #get [alpha, alpha_lo, alpha_hi]
    alphas = bca_pl(d,dmin,dmax, alphas_boot[alphas_boot >= min_alpha], ci = bca_ci, stepsize = stepsize)
    
    #get [snz, snz_lo, snz_hi]
    snzs = bca_fit(s,d,smin,smax, snzs_boot[alphas_boot >= min_alpha], ci = bca_ci)
    
    #get [lhs, lhs_lo, lhs_hi]
    lhss = bca_lhs(s,d,smin,smax,dmin,dmax,lhss_boot[alphas_boot >= min_alpha],ci = bca_ci,ystepsize=stepsize)
    
    #get [rel, rel_lo, rel_hi]
    rels = bca_rel(s,d,smin,smax,dmin,dmax,lhss_boot[alphas_boot >= min_alpha],snzs_boot[alphas_boot >= min_alpha],ci=bca_ci,ystepsize = stepsize)
    
    
    return taus, alphas, snzs, lhss, rels