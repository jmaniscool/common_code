# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:55:22 2024

@author: sickl
"""

#Add all defunct functions that are not in use any more.

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

#lognormal likelihood that does not appropriately deal with the boundaries.
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

#get the (non-bca) confidence intervals from the array of bootstrapped values.
#95% confidence interval is default. That is, 95% of values are going to be in range
#(lo,hi)
def confidence_intervals(vals,ci = 0.95):
    ci = 100 - ci
    mu = np.nanmedian(vals)
    lo = np.nanpercentile(vals,ci/2)
    hi = np.nanpercentile(vals,100-ci/2)
    return mu, lo, hi

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

#find the power law index "exactly" using the zero of the derivative of the log likelihood function.
#Works more quickly and over a broader range than scipy.minimize version.
#ASSUME X IS SORTED AND IT GOES FROM XMIN TO XMAX INCLUSIVE

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

#attempt jit acceleration. Ensure x geq xmin and x leq xmax is true before using.
@numba.njit
def pl_like_fast(x,xmin,xmax,alpha):
    ll = 0
    X = xmax/xmin
    dist = np.log(((alpha-1)/xmin)*(1/(1-X**(1-alpha)))*(x/xmin)**(-alpha))
    ll = sum(dist)
    return ll, dist

#Agrees with MLE solution to "Truncated Pareto" distribution from Table 1 of https://pearl.plymouth.ac.uk/bitstream/handle/10026.1/1571/2013Humphries337081phd.pdf?sequence=1
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
    def f(alpha):
        #added edge case for alpha near 1
        #test for alpha = 1
        if alpha == 1:
            val = -ln(ln(xmax/xmin)) - S/n #equation from Deluca & Corrall 2013, equation 12.
            if val < 0: #in some cases (i.e. in when the data is pulled from an odd lognormal distribution), the sum will be negative. In those cases, return a random positive value.
                return 100
            return val
        #large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
        test_xmin = np.log10(xmin)*(-alpha+1)
        if test_xmin > 100:
            return -1e12
        
        #if the tested alpha is very low, use a taylor approximation
        if alpha < 1 + 1e-7:
            y = alpha-1
            beta = y*ln(xmax/xmin)
            gam = ln(xmax/xmin) - y*ln(xmax)*ln(xmax) + y*ln(xmin)*ln(xmin)
        else:
            beta = -xmax**(-alpha+1) + xmin**(-alpha+1)
            gam = xmax**(-alpha+1)*ln(xmax) - xmin**(-alpha+1)*ln(xmin)
            
        y = n/(alpha - 1) - S - n*(gam/beta)
        
        return y

    
    #using SciPy optimize. Slower.
    #out = scipy.optimize.root_scalar(f, bracket = (1, 20))
    
    #using numba jitted brent's method for finding the root. About 10x faster than scipy.
    alpha = brent_findmin(x)
    
    ll = pl_like(x,x[0],x[-1],alpha)[0]
    return alpha,ll

#wrapper for find_pl which does not assume x is sorted and is in the range of xmin and xmax.
#replace with brent_findmin
def find_pl_exact(x,xmin,xmax = 1e6):
    x = arr(x)
    tmp = x[(x >= xmin)*(x <= xmax)]
    alpha = brent_findmin(tmp)
    #alpha,ll = find_pl_exact_sorted(np.sort(tmp))
    ll = pl_like(x,xmin,xmax,alpha)[0]
    return alpha,ll

#v2 use minimize scalar to be about 3-4x faster than nelder-mead.
def find_pl_fast(x,xmin,xmax = 1e6):
    if len(x) == 1:
        return np.nan, -1e12
    
    
    #mymean = lambda a: -pl_like_fast(x,xmin,xmax,a)[0]
    #myfit = optimize.minimize(mymean,2,method = 'Nelder-Mead',bounds = [(1,1e6)])
    #myfit = optimize.minimize_scalar(mymean, bounds = (1,30))
    alpha = brent_findmin(x)    
    ll = pl_like_fast(x,x[0],x[-1],alpha)[0]
    return alpha,ll

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

#I would not recommend using this right now -- it appears not to give us answers as significant when we indeed believe them to be.
def tost(valsa,valsb,sig = 0.2):
    conf = 0 #level of confidence of determination.
    dist = valsa-valsb #construct the distribution of differences
    med,lo,hi = confidence_intervals(dist,ci = 68) #get median value med, with 68% CI = (lo,hi)
    lo = med - lo
    hi = hi - med #convert CI to difference from median
    zlo = (abs(med) + sig)/lo
    zhi = (abs(med) - sig)/hi
    
    z = min(abs(zlo),abs(zhi))
    
    p = scipy.stats.norm.sf(z)

    return z,p

#method of moments estimator of mu, sigma for truncated normal distribution. Broken as of Jul 5, 2024.
def momsolve(data,a,b):
    n = len(data)
    def f(p):
        mu = p[0]
        sig = p[1]
        #PDFs
        phib = normpdf((b-mu)/sig)
        phia = normpdf((a-mu)/sig)
        
        phix = normpdf((data-mu)/sig)
        x = (data-mu)/sig

        #CDFs
        Phib = normcdf((b-mu)/sig)
        Phia = normcdf((a-mu)/sig)
        eq1 = mu - sig*((phib-phia)/(Phib - Phia)) - np.sum(x*phix)
        eq2 = mu*mu + sig*sig - sig*(((mu+b)*phib - (mu+a)*phia)/(Phib - Phia)) - np.sum(x*x*phix)
        return eq1,eq2
    
    muinit,siginit = scipy.optimize.fsolve(f,[1,1])
    return muinit,siginit

##DEFUNCT FUNCTIONS

"""

#defunct lognormal functions from powerlaw() package that did not accurately include upper truncation.
#the much slower version that's more accurate using mpmath. Not recommended.
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

#attempt a much faster version of lognormal_like to help with fitting.
#can be jitt-ed if required for speed, though optimize_minimize will need to be rewritten for that and this would have diminishing returns.
def lognormal_like_fast(x, xmin, xmax, mu, sigma):
    import math
    
    ##  NOTE: the corresponding log-likelihood function used by the powerlaw() library
    #   does not appropriately limit the boundaries for mu or sigma. Mu can, in
    #   principle, be any value. Negative values of mu might be expected if the generative process is from
    #   multiplication of many positive random variables, for instance. This is possible in our system and AGNs, so
    #   we should not limit mu or sigma. While AGN and stars are very different systems, the MHD equations
    #   should still apply in both cases, albeit in different limits.
    
    #catch the illegal values of mu and sigma
    #if sigma <= 0 or mu < log(xmin):
    #    return -1e12, np.zeros(len(x))
    x = x[(x >= xmin)*(x <= xmax)]
    #mpmath is used because it has higher accuracy than scipy
    #dist = -mylog(x)-((mylog(x)-mu)**2/(2*sigma**2)) + 0.5*mylog(2/(pi*sigma**2))- mylog(myerfc((mylog(xmin)-mu)/(mysqrt(2)*sigma)))
    
    #use numpy and math module because accuracy is not as important as speed for our application.
    dist = -np.log(x)-((np.log(x)-mu)**2/(2*sigma**2)) + 0.5*np.log(2/(np.pi*sigma**2))- np.log(math.erfc((np.log(xmin)-mu)/(np.sqrt(2)*sigma)))
    
    ll = np.sum(dist)
    
    return ll, dist

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

#fast version of find_lognormal which relies on faster lognormal implementation that works OK for reasonable data.
def find_lognormal_fast(x,xmin,xmax = 1e6):
    x = np.array(x)
    logx = np.log(x[(x >= xmin)*(x <= xmax)])
    initial_guess = [np.mean(logx),np.std(logx)]
    mymean = lambda par: -lognormal_like_fast(x,xmin,xmax,par[0],par[1])[0]
    opt_results = optimize.minimize(mymean,initial_guess,method = 'Nelder-Mead')
    mu = opt_results.x[0]
    sigma = opt_results.x[1]
    ll = -opt_results.fun
    return mu,sigma,ll

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
