#Alan Long 5/23/19
#Last edited 9/9/19
#Vectorized by Jordan 12/19/23

#Jordan additions 12/20/23
"""
-Added options for pointwise and simultaneous CCDF errorbars.
    --Pointwise approach will give you bounds on each point such that i.e. 95% of the time the data plus those errorbars will be on the true CCDF.
        --Pointwise approach based on frequentist Clopper-Pearson approach for estimating confidence intervals on the binomial distribution https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=1132&context=jmasm
        --Verified with examining dispersion of CDFs at each point.
    --Simultaneous approach will give bounds that the extrema of the CCDF will be within the range of the true CCDF i.e. 95% of the time.
        --Simultaneous approach based on Dvoretzky–Kiefer–Wolfowitz inequality which says the true CDF will lay in a range of +-sqrt(log(2/alpha)/(2*n)) of the empirical CDF with alpha confidence (alpha = 0.05 for 95% CI)
        --Verified with bootstrapping of CDFs
        --See for more details: https://en.wikipedia.org/wiki/Dvoretzky%E2%80%93Kiefer%E2%80%93Wolfowitz_inequality

-Original approach, Nir's approach, gives pointwise Bayesian confidence intervals.
    --Nir's approach assumes the data are treated as a "bag" from which data can be drawn randomly to estimate the current value of the errorbar.
    --Performs at basically the same level as the pointwise approach, though some future work might be done to consider which approach is technically correct for our applications.
    
Citations:
    [1] Nir Friedman's thesis for Nir's method
    [2] For Clopper-Pearson pointwise confidence intervals, see https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=1132&context=jmasm
    [3] For Dvoretzky-Kiefer-Wolfowitz simultaneous confidence intervals, see https://projecteuclid.org/journals/annals-of-probability/volume-18/issue-3/The-Tight-Constant-in-the-Dvoretzky-Kiefer-Wolfowitz-Inequality/10.1214/aop/1176990746.full
    
Further reading:
    https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0258(19980430)17:8%3C857::AID-SIM777%3E3.0.CO;2-E
    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2004JB003479

"""




import numpy as np
from scipy.special import betainc
from scipy.optimize import minimize
from scipy.optimize import root_scalar
import scipy
from .get_ccdf_arr import ccdf as ccdf


#Errorbar handling wrapper. Returns relative errorbar around each point. Default to pointwise.
def ccdf_errorbar(data, ci = 0.95, method = 'pointwise'):
    """
    CCDF errorbar wrapper. Returns the y error of the CCDF, [lo, hi] which are the
    (100*ci)-percentile confidence intervals at the bottom and top respectively.
    That is, for cx,cy = cc.ccdf(data), the (100*ci)% confidence interval extends
    from cy - lo to cy + hi.
    
    This function calculates three types of confidence intervals, explained briefly below.
    Generally, prefer to use the simultaneous approach since that models the shape of the
    underlying CCDF rather than the dispersion that each individual point can obey.
    
    'simultaneous':
        The Dvoretzky–Kiefer–Wolfowitz inequality (DKW) upper bound confidence interval
        on the shape of the CCDF. In other words, if N synthetic datasets are generated
        via bootstrapping on the data, ci*N runs will have shapes within thes DKW
        intervals. The DKW confidence interval is symmetric about the CCDF. This should
        be used when estimating the shape of the CCDF (i.e. when calculating the scaling
        regime xmin/xmax via monte carlo.).
    
    'pointwise':
        The frequentist Clopper-Pearson confidence bands which can be asymmetric and
        are calculated using the root of a beta function. Calculates the range over
        which each individual data point can vary from the empirical distribution
        function. That is, e.g. 5% of the time the empirical CDF will lie outside
        of this range for each data point. Or, more roughly, 5% of the data will lie
        outside of this range on each synthetic run. For more information, see
        https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        
    'nir':
        The Bayesian pointwise confidence bands which can be asymmetric and are
        calculated using the root of a beta function. The only significant difference
        between this and the pointwise approach is the fact that the pointwise
        approach is calculated from a frequentist approach whereas this is calculated
        using a Bayesian approach (leading to small indexing differences). For more
        information, see Nir Friedman's thesis.

    Parameters
    ----------
    data : float array
        The data to calculate the CCDF errorbar around.
    ci : float, optional
        The confidence interval to return, 0 < ci < 1. The default is 0.95 for 95% CI.
    method : string, optional
        The type of confidence interval to calculate.
        The default is 'pointwise'.

    Returns
    -------
    bot: float
        The lower confidence interval, cy - lo.
    top: float
        The upper confidence interval, cy + hi

    """
    cx,cy = ccdf(data)
    datalen = len(data)
    if method == "simultaneous":
        fun = simultaneous_errorbars
    elif method == "pointwise":
        fun = pointwise_errorbars #frequentist approach which does not require prior.
    elif method == "nir":
        fun = nir_errorbars #bayesian approach with uniform priors
    else:
        print("ERROR: Nethod must be either simultaneous, pointwise, or nir. Returning.")
        return np.array([-1]), np.array([-1])            
    return fun(cy, ci, datalen)

#Calculates the frequentist Clopper-Pearson pointwise confidence band for eacof the CCDF.
#Use when figuring the dispersion each point may have individually from
#the true CCDF and for collapses.
def pointwise_errorbars(histy, ci, datalen):
    """
    Calculates the frequentist Clopper-Pearson pointwise confidence band for each
    datapoint in the CCDF. Use when the dispersion of the data away from the
    empirical distribution function should be known -- this is a niche use case
    and should be avoided, noting that (100*(1-ci))% of the time the data will lie
    outside of this bound in synthetic datasets.

    Parameters
    ----------
    histy : float array
        The y values of the CCDF, the output of the ccdf() function.
    ci : float
        The confidence interval to calculate the error bars over.
    datalen : int
        The length of the data. len(histy) != datalen because there is a non-zero
        chance of overlaps in all data.

    Returns
    -------
    bot: float
        The lower confidence interval, cy - lo.
    top: float
        The upper confidence interval, cy + hi

    """
    ci_bot = (1 - ci)/2
    ci_top = (1 + ci)/2
    
    N = datalen #fix so the data length is appropriately accounted for.
    
    num_successes = (1-histy)*N #Convert from CCDF to number of trials less than (or equal to if using scipy option) current number.
    
    errs_top = np.zeros(len(histy))
    errs_bot = np.zeros(len(histy))
    for i in range(len(histy)):
        n = num_successes[i]
        
        #if n = 0 or N, give the exact solutions.
        if n == 0:
            errs_bot[i] = 0
            errs_top[i] = 1 - (ci_bot/2)**(1/N)            
        elif n == N:
            errs_bot[i] = (ci_bot/2)**(1/N)
            errs_top[i] = 1
        else:
            errs_top[i] = root_scalar(beta_hi_pearson, args = (N,n, ci_top), bracket = [0,1]).root
            errs_bot[i] = root_scalar(beta_lo_pearson, args = (N,n, ci_bot), bracket = [0,1]).root
        

    #convert to error on CCDF
    top = 1 - errs_bot
    bot = 1 - errs_top
        
    #convert to relative error
    top = top - histy
    bot = histy - bot
        
    return bot, top

def beta_lo_pearson(x,N,n_success,err):
    v = betainc(n_success,N - n_success + 1,x)
    return v - err
    #return v*v-2*err*v + err*err #minimizing square of betainc function instead of abs(betainc)
    #return abs(v - err)
    
def beta_hi_pearson(x,N,n_success,err):
    v = betainc(n_success + 1,N - n_success,x)
    return v - err
    #return v*v - 2*err*v + err*err #minimizing square of betainc function instead of abs(betainc)        
    


#calculate the simultaneous CCDF error bars. Use when estimating the possible
#dispersion the empirical CCDF may have from the true CCDF 95% of the time.
#This can be useful when fitting to an expected distribution, i.e. if you are trying to prove that a distribution could be pulled from a power law of particular exponent.
def simultaneous_errorbars(histy, ci, datalen):
    """
    Calculates the symmetric DKW errorbars. This should be preferred when calculating
    properties related to the shape of the distribution, i.e. obtaining xmin/xmax,
    obtaining the underlying exponent value.

    Parameters
    ----------
    histy : float array
        The y values of the CCDF, the output of the ccdf() function.
    ci : float
        The confidence interval to calculate the error bars over.
    datalen : int
        The length of the data. len(histy) != datalen because there is a non-zero
        chance of overlaps in all data.

    Returns
    -------
    bot: float
        The lower confidence interval, cy - lo.
    top: float
        The upper confidence interval, cy + hi

    """
    alpha = 1-ci
    N = datalen
    epsilon = np.sqrt(np.log(2/alpha)/(2*N))
    
    #relative errorbars
    errs_top = epsilon*np.ones(len(histy))
    errs_bot = epsilon*np.ones(len(histy))    
    errs_top[histy + epsilon > 1] = (1-histy)[histy + epsilon > 1]
    errs_bot[histy - epsilon < 0] = histy[histy - epsilon < 0]
    
    return errs_bot, errs_top


#Jordan version of Nir's Bayesian pointwise errorbar approach. Use ci = 0.95 to get 95% CI.
def nir_errorbars(histy, ci, datalen):

    histy = np.array(histy)
    N = datalen
    errs_top = np.zeros(N)
    errs_bot = np.zeros(N)
    ci_bot = (1 - ci)/2
    ci_top = (1 + ci)/2
    
    num_successes = (1-histy)*N #Convert from CCDF to number of trials less than (or equal to if using scipy option) current number.
    
    
    for i in range(N):
        n = num_successes[i]
        
        #using root_scalar to solve function (faster than scipy)
        errs_top[i] = root_scalar(beta_fun_nir, args = (N,n, ci_top), bracket = [0,1]).root
        errs_bot[i] = root_scalar(beta_fun_nir, args = (N,n, ci_bot), bracket = [0,1]).root
        
        #point probability function approach. Less sensitive to solver, but slower.
        #errs_bot[i], errs_top[i] = scipy.stats.beta.interval(ci,n+1,N-n+1) 
        
        
        #incomplete beta function approach using minimize.
        #Approximate and sensitive to solver.
        """
        mybeta_top = lambda x: beta_fun(x,N,i + 1,ci_top)
        mybeta_bot = lambda x: beta_fun(x,N,i + 1,ci_bot)
        #print(minimize_scalar(mybeta_top, bounds = (0,1)).x)
        errs_top[i] = minimize_scalar(mybeta_top, bounds = (0,1), method = 'bounded').x#, method = 'Nelder-Mead').x
        errs_bot[i] = minimize_scalar(mybeta_bot, bounds = (0,1), method = 'bounded').x#, method = 'Nelder-Mead').x
        """
    
    #convert to CCDF errorbars
    top = 1 - errs_bot
    bot = 1 - errs_top
        
    #convert to relative error
    top = top - histy
    bot = histy - bot
    
    #fix errorbars to be within 0 <= p <= 1
    bot[bot < 0] = 0
    top[top < 0] = 0
    bot[histy-bot < 0] = histy[histy - bot < 0]
    top[histy + top > 1] = (1-histy)[histy + top > 1]
    #print('foo')
    
    return bot, top

def beta_fun_nir(x,N,n_success,err):
    v = betainc(n_success + 1,N - n_success + 1,x)
    return v - err



    
    

#Alan version

#this program makes error bars for your ccdfs. It accepts the histY from
#GetCumDist and outputs two arrays of the same length which are your error
#bar values for top and bottom. It defaults to 2sigma. This is based on the error analysis in
#Nir's thesis apendix B8

from scipy.optimize import minimize_scalar
def ccdf_errorbars_alan(histY1,sigmas1):
    histY = histY1.copy()
    sigmas = sigmas1
    global N
    N=len(histY)
    errs_top=[]
    errs_bot=[]

    global i
    for i in range(1,N+1):
        #now find the zeros
        
        #NOTE: minimize_scalar for abs(beta - err) is poorly behaved.
        #Instead, find the zero of beta - err using root_scalar.
        res=minimize_scalar(beta_hi,bounds=(0,1),method='bounded')
        errs_top.append(float(res.x))
        res=minimize_scalar(beta_lo,bounds=(0,1),method='bounded')
        errs_bot.append(float(res.x))

    #finally find the difference to get the error bars
    errs_top=np.subtract(errs_top,histY)
    errs_bot=np.subtract(histY,errs_bot)
    
    #Jordan note: I don't think it's necessary to do this step, so it is not included in the more updated versions of the code above.
    errs_top=histY*(np.exp(errs_top/histY)-1)
    errs_bot=histY*(-np.exp(-errs_bot/histY)+1)
    return errs_top,errs_bot

#define the functions, with the disired values taken off

def beta_hi(x1):
    x = x1.copy()
    return abs(betainc(N-i+1,i+1,x)-.975)#change the .975 to desired sigma as needed
def beta_lo(x1):
    x = x1.copy()
    return abs(betainc(N-i+1,i+1,x)-.025)#change .025 same as .975