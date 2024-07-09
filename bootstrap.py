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
from .likelihoods import find_pl, find_tpl

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
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]



#get a new random xmin_star and xmax_star which varies over -dex to dex in log scale and ensures xmin < xmax.
#note that ensuring xmin > xmax makes the "random" value of xmax covariant with xmin.
@numba.njit
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
@numba.njit
def loginterp(x,y,xmin,xmax, bins = 50):
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
@numba.njit
def bootstrap_core(s,d,smin, smax, dmin, dmax,vm, vmin, vmax,logs,logd,logvm, fun, dex, ctr_max):
    
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
    tau = fun(scc,smin_star,smax_star)[0]
    alpha = fun(dcc, dmin_star,dmax_star)[0]
    
    #find snz and (tau-1)/(alpha-1)
    #snz = scipy.stats.linregress(logscc,logdcc).slope
    snz = fit_poly(logscc,logdcc,1)[0]
    sdlhs = np.nan
    if alpha > 1:
        sdlhs = (tau-1)/(alpha-1)
    
    #if vm is given, also calculate velocity statistics.
    if (vmin != 1)*(vmax != 1):
        mu = fun(vmcc,vmin_star,vmax_star)[0]
        #sp = scipy.stats.linregress(logscc,logvmcc).slope
        #pnz = scipy.stats.linregress(logdcc,logvmcc).slope
        sp = fit_poly(logscc,logvmcc,1)[0]
        pnz = fit_poly(logdcc,logvmcc,1)[0]
        svlhs = (tau-1)/(mu-1)
        dvlhs = (alpha-1)/(mu-1)
    
    
    return tau,alpha,mu, sdlhs,svlhs,dvlhs, snz,sp,pnz

@numba.njit(parallel = True)
def bootstrap_parallel(num_runs,s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,fun,dex,ctr_max):
    vals = np.zeros((num_runs,9))
    for i in numba.prange(num_runs):
        tmp = bootstrap_core(s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,fun,dex,ctr_max)
        vals[i] = tmp
    
    return vals.transpose()

#v2 of bootstrap, updated Feb 27, 2024. Written to take advantage of various programming fundamentals improvements Jordan learned since the original bootstrap was written
def bootstrap(s,d, smin, smax, dmin, dmax, vm = None, num_runs = 10000, dex = 0.25, ctr_max = 10, min_events = 10):
    
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
    
    fun = find_pl #set fun to be the power_law() function
    
    #do the bootstrapping (serial). A bit slower than the original bootstrapping approach.
    #for i in range(num_runs):
    #    if i % 1000 == 0:
    #        print(i)
    #    taus[i],alphas[i],mus[i],sdlhss[i],svlhss[i],dvlhss[i],snzs[i],sps[i],pnzs[i] = bootstrap_core(s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,fun,dex,ctr_max)
    vals = bootstrap_parallel(num_runs,s,d, smin,smax,dmin,dmax,vm,vmin,vmax,logs,logd,logvm,fun,dex,ctr_max)
    taus = vals[0,:]
    alphas = vals[1,:]
    mus = vals[2,:]
    sdlhss = vals[3,:]
    svlhss = vals[4,:]
    dvlhss = vals[5,:]
    snzs = vals[6,:]
    sps = vals[6,:]
    pnzs = vals[7,:]
        
    return taus,alphas,mus, sdlhss,svlhss,dvlhss, snzs,sps,pnzs

#the core of the BCa correction given theta_hat and jackknifed samples theta_jk. Validated against scipy.stats.bootstrap.
def bca_core(theta_hat,theta_jk, bootstrap_estimates, alpha):
    
    from matplotlib import pyplot as plt
    
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
def bca_pl(x, xmin,xmax, bootstrap_estimates, ci = 0.95):    
    alpha = 1-ci #get alpha (e.g. confidence interval of 95% means alpha = 0.05)
    
    #compute the "true" value of theta
    xc = x[(x >= xmin)*(x <= xmax)]
    theta_hat = find_pl(xc,xmin,xmax)[0]
    
    #Jackknife estimation of acceleration (from Scipy)
    numdat = len(xc)
    theta_jk = np.zeros(numdat)
    for i in range(numdat):
        theta_jk[i] = find_pl(np.delete(xc,i),xmin,xmax)[0]

    #calculate
    
    if (theta_hat == 1) or (np.sum(theta_jk) == numdat):
        return theta_hat, np.nan, np.nan
    
    return bca_core(theta_hat,theta_jk, bootstrap_estimates, alpha)

#compute the BCA confidence intervals for combined statistics. i.e. like (tau-1)/(alpha-1)
def bca_lhs(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates, ci = 0.95):
    alpha = 1-ci
    
    xc = x[(x >= xmin)*(x <= xmax)]
    yc = y[(y >= ymin)*(y <= ymax)]
    
    thetax_hat = find_pl(xc,xmin,xmax)[0]
    thetay_hat = find_pl(yc,ymin,ymax)[0]

    #get the mean value of (tau-1)/(alpha-1)
    theta_hat = (thetax_hat - 1)/(thetay_hat - 1)
    
    nx = len(xc)
    ny = len(yc)
    
    #The real way would be to jackknife over avalanches.
    #That is, remove avalanches (1) one by one (remove same index i,j), (2) compute xc and yc from xmin, xmax, ymin, ymax, (3) compute tau, alpha from the subsample
    
    #assuming samples of x and y are independent (they're not! But the resampling in bootstrapping assumes s,d,etc are independent)
    #jackknife over thetax
    thetax_jk = np.zeros(nx)
    for i in range(nx):
        thetax_jk[i] = find_pl(np.delete(xc,i),xmin,xmax)[0]
    
    #jackknife over thetay
    thetay_jk = np.zeros(ny)
    for i in range(ny):
        thetay_jk[i] = find_pl(np.delete(yc,i),ymin,ymax)[0]   
    #get the total jackknife (??)
    theta_jk = np.zeros(nx*ny)
    for i in range(nx):
        for j in range(ny):
            theta_jk[int(j*nx + i)] = (thetax_jk[i] - 1)/(thetay_jk[j] - 1) #compute (tau-1)/(alpha-1) for all jackknifed nx and ny, taking advantage of independence of x and y
        

    if (theta_hat == 1) or (np.sum(theta_jk) == nx*ny):
        return theta_hat, np.nan, np.nan

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


    if (theta_hat == 1) or (np.sum(theta_jk) == nx):
        return theta_hat, np.nan, np.nan
        
    return bca_core(theta_hat,theta_jk,bootstrap_estimates,alpha)

#BCa to determine the exponent relationship confidence intervals, i.e. (tau-1)/(alpha-1) = snz
#HOW TO USE: pass in bca_rel(s,d,smin,smax,dmin,dmax,lhss,snzs, ci = 0.95) to get bootstrapping.
def bca_rel(x,y,xmin,xmax,ymin,ymax,bootstrap_estimates,bootstrap_estimates2, ci = 0.95):
    
    from matplotlib import pyplot as plt
    alpha = 1 - ci
    
    xc = x[(x>= xmin)*(x <= xmax)]
    yc = y[(y >= ymin)*(y <= ymax)]
    
    logxc = np.log10(xc)
    logyc = np.log10(y[(x >= xmin)*(x <= xmax)])

    fit_hat = scipy.stats.linregress(logxc,logyc).slope
    thetax_hat = find_pl(xc,xmin,xmax)[0]
    thetay_hat = find_pl(yc,ymin,ymax)[0]
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
        thetax_jk[i] = find_pl(tmpx,xmin,xmax)[0]
        thetafit_jk[i] = scipy.stats.linregress(tmplogx,tmplogy).slope
    
    #jackknife over thetay
    thetay_jk = np.zeros(ny)
    for i in range(ny):
        thetay_jk[i] = find_pl(np.delete(yc,i),ymin,ymax)[0]
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



#get the (non-bca) confidence intervals from the array of bootstrapped values.
#95% confidence interval is default. That is, 95% of values are going to be in range
#(lo,hi)
def confidence_intervals(vals,ci = 0.95):
    ci = 100 - ci
    mu = np.nanmedian(vals)
    lo = np.nanpercentile(vals,ci/2)
    hi = np.nanpercentile(vals,100-ci/2)
    return mu, lo, hi