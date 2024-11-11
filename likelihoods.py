"""
Created Jul 05 2022 (Jordan Sickle)
Updated Nov 11 2024 (Ethan Mullen)

-- Performs the analysis of Clauset et al 2009, with some methodology, functions, and inspiration taken from
the python powerlaw() package by Alstott et al 2014.
    -- PLEASE CITE BOTH OF THESE SOURCES WHEN USING THIS PACKAGE!
    -- [1] J. Alstott, E. Bullmore, D. Plenz, PLoS One 9 (2014) 85777.
    -- [2] A. Clauset, C.R. Shalizi, M.E.J. Newman, Http://Dx.Doi.Org/10.1137/070710111 51 (2009) 661–703.

-- The power law PDF and error bars for the exponents are estimated using Mo's method
    -- The error is taken as the range of exponent values found when the xmin and xmax boundaries are varied by +/- 0.25 decades.
    -- This is a purposeful overestimate of the error, in the event that bootstrapping gives errors that seem too low.

-- For more details, please read:
    -- [3] M.A. Sheikh, R.L. Weaver, K.A. Dahmen, Phys. Rev. Lett. 117 (2016).
    -- [4] M.A. Sheikh, Avalanches in Stars and at Finite Temperature (2019).
    -- [5] N. Friedman, Avalanches in disordered systems (2013).


-- QUICKSTART GUIDE:
    -- TO TEST WHICH DISTRIBUTION IS MOST LIKELY:
        -- Step 1: Obtain the A^2 values from the ad() test on sizes and durations.
            -- Distributions with lowest A^2 are most consistent with the data.
        -- Step 2: AND/OR use the LLR test contained in the llr_wrapper() function.
            -- If llr > 0 and p <= p_crit (for some chosen p_crit), then the first function is preferred.
            -- If llr < 0 and p <= p_crit (for some chosen p_crit), then the second function is preferred.
            -- Otherwise, neither is statistically preferred.
            -- I choose p_crit = 0.01 for my analysis, but this is a quite stringent boundary I've found.
            -- Please note the LLR test only works when comparing two distributions created from the same number of data!!

    -- TO OBTAIN ERRORBARS ON THE POWER LAW:
        -- Step 1: Use out = bootstrap() to obtain histograms which explain the variance of all exponents.
        -- Step 2: Obtain confidence intervals from the confidence_intervals() function.
            -- Default is 95% CI
            -- Example: med, hiCI, loCI = confidence_intervals(out[0]) gives median value med, 95% CI = (loCI, hiCI)
        -- Step 3: Test the exponent relationships using tost().
            -- Most stars pass tost with p < 0.1 when sig is set to, like, 0.2.

-- Generator functions:
    -- generate_test_data_with_xmax()
        -- Updated Nov 11 2024 by Ethan
        -- Test finding data with xmin and xmax, where power law scaling is noted between xmin and xmax and goes exponential elsewhere.
    -- pl_gen()
        -- Produces a power law between xmin and xmax when supplied with a vector of random variables with elements 0 < r < 1
    -- pl_gen_discrete()
        -- Produces a power law between xmin and xmax when supplied with a vector of random variables with elements 0 < r < 1
    -- lognormal_gen()
        --
    -- tpl_gen()
        -- Gives a truncated power law with given alpha and lambda.
    -- exp_gen()
        -- Produces an exponentially distributed PDF with a given xmin.

-- Likelihood functions:
    -- pl_like()
        --
    -- pl_like_discrete()
        --
    -- exp_like()
        --
    -- tpl_like()
        --
    -- lognormal_like_truncated()
        --
    -- lognormal_like()
        --

-- Maximum likelihood estimation (MLE) functions
    -- find_pl():
        --
    -- find_pl_discrete()
        --
    -- find_tpl()
        --
    -- find_exp()
        --
    -- momsolve_truncnorm()
        --
    -- find_trunc_norm()
        --
    -- find_lognormal_from_normal()
        --
    -- find_lognormal_truncated()
        --
    -- find_lognormal()
        --

-- Log-likelihood ratio (LLR) comparisons
    -- llr()
        --
    -- llr_wrap()
        --

-- Anderson-Darlin (AD) test
    -- ad()
        --
"""
# My packages
from .brent import brent_findmin, brent_findmin_discrete

# Python packages
import mpmath as mp
from mpmath import gammainc
import numpy as np
import scipy
import time
import numba

# Vectorized operations.
# Not really required, since we can get good precision with scipy.
mylog = np.vectorize(mp.log)
myexp = np.vectorize(mp.exp)
mysqrt = np.vectorize(mp.sqrt)
myerfc = np.vectorize(mp.erfc)
myround = np.vectorize(round)
myfloat = np.vectorize(float)
myint = np.vectorize(int)

ln = np.log
exp = np.exp
pi = np.pi
sqrt2 = np.sqrt(2)
arr = np.array

lognormlogpdf = scipy.stats.lognorm.logpdf
lognormcdf = scipy.stats.lognorm.cdf

normlogpdf = scipy.stats.norm.logpdf
normcdf = scipy.stats.norm.cdf
normpdf = scipy.stats.norm.pdf

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GENERATOR FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


@numba.njit
def generate_test_data_with_xmax(datasize, xmin, xmax, alpha):
    """
    Enforced continuity at boundaries of xmin and xmax.

    Parameters
    ----------
    datasize
    xmin
    xmax
    alpha

    Returns
    -------

    """
    x = np.random.rand(datasize)

    lam = xmin ** (-alpha + 1) * (np.exp(alpha) - 1) / alpha
    eta = (xmin ** (-alpha + 1) - xmax ** (-alpha + 1)) / (alpha - 1)
    gam = xmax ** (-alpha + 1) / alpha

    co = 1 / (lam + eta + gam)

    outs = np.zeros(len(x))

    c_star_lo = lam * co
    c_star_hi = (lam + eta) * co

    for i in range(len(x)):
        if x[i] < c_star_lo:
            outs[i] = (xmin / alpha) * ln(exp(alpha) / (exp(alpha) - (alpha / co) * xmin ** (alpha - 1) * x[i]))
        elif x[i] < c_star_hi:
            outs[i] = (xmin ** (-alpha + 1) + (alpha - 1) * (lam - x[i] / co)) ** (1 / (-alpha + 1))
        else:
            outs[i] = xmax - (xmax / alpha) * ln(1 - ((x[i] / co) - lam - eta) * xmax ** (alpha - 1) * alpha)

    return outs


@numba.njit
def pl_gen(datalen, xmin, xmax, alpha):
    """
    Produces a power law between xmin and xmax when supplied with a vector of random variables with elements 0 < r < 1

    Parameters
    ----------
    datalen
    xmin
    xmax
    alpha

    Returns
    -------

    """
    x = np.random.rand(datalen)
    X = xmax / xmin
    out = (1 - x * (1 - X ** (1 - alpha))) ** (1 / (1 - alpha)) * xmin

    return out


@numba.njit
def pl_gen_discrete(datalen, xmin, xmax, alpha):
    """
    Generate a set of discrete data that obey a power law relationship, using
    the approximation given in Clauset et al 2009 just above equation D6.

    Sources
    [1] Clauset et al 2009. Power-Law Distributions in Empirical Data.

    Parameters
    ----------
    datalen : int
        The length of the data.
    xmin : int
        The minimum value of the scaling regime, x >= xmin. Clauset et al 2009
        claim that this approximation is good to, at worst, 10% for xmin = 1
        but in practice is less than 1% when xmin = 10.
    xmax : int
        The maximum value of the scaling regime, x <= xmax.
    alpha : float
        The underlying power-law index in the data.

    Returns
    -------
    out : array of ints
        Data that are (approximately) power-law distributed in discrete x.

    """

    x = np.random.rand(datalen)
    xmin_p = xmin - 0.5
    xmax_p = xmax - 0.5
    X = xmax_p / xmin_p
    out = np.floor((1 - x * (1 - X ** (1 - alpha))) ** (1 / (1 - alpha)) * xmin_p + 0.5)

    return out


def lognormal_gen(size, a, b, mu, sigma):
    """
    Much faster and accurate enough generator for lognormal data.

    Parameters
    ----------
    size
    a
    b
    mu
    sigma

    Returns
    -------

    """
    norm = scipy.stats.truncnorm.rvs((np.log(a) - mu) / sigma, (np.log(b) - mu) / sigma, loc=mu, scale=sigma,
                                     size=int(size))
    lognorm = exp(
        norm)  # Because "norm" is normally distributed between log(a) and log(b), exp(norm) will be lognormally distributed between a and b.

    return lognorm


def tpl_gen(r, xmin, alpha, Lambda):
    """
    From powerlaw() library.
    Gives a truncated power law with given alpha and lambda.
    Can probably be sped up significantly.

    Parameters
    ----------
    r
    xmin
    alpha
    Lambda

    Returns
    -------

    """

    def helper(r):
        from numpy import log
        from numpy.random import rand
        st = time.time()
        dt = time.time() - st
        while dt <= 0.1:  # Spend at most 0.1 seconds per data point
            dt = time.time() - st
            x = xmin - (1 / Lambda) * log(1 - r)
            p = (x / xmin) ** -alpha
            if rand() < p:
                return x
            r = rand()

        # If spent too much time, then return nan
        return np.nan

    from numpy import array

    return array(list(map(helper, r)))


def exp_gen(size, xmin, Lambda):
    """
    Produces an exponentially distributed PDF with a given xmin.

    Parameters
    ----------
    size
    xmin
    Lambda

    Returns
    -------

    """
    x = np.random.rand(size)
    out = -np.log(1 - x) / Lambda
    out = out + xmin

    return out


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LIKELIHOOD FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


@numba.njit
def pl_like(x, xmin, xmax, alpha):
    """
    Attempt jit acceleration.

    Parameters
    ----------
    x
    xmin
    xmax
    alpha

    Returns
    -------

    """
    x = x[(x >= xmin) * (x <= xmax)]

    if alpha == 1:
        return -1e-12, np.zeros(len(x))

    X = xmax / xmin
    dist = np.log(((alpha - 1) / xmin) * (1 / (1 - X ** (1 - alpha))) * (x / xmin) ** (-alpha))
    ll = np.sum(dist)

    return ll, dist


@numba.njit
def pl_like_discrete(normx, normxmin, normxmax, alpha):
    """
    The discrete version of the powerlaw likelihood function.

    Parameters
    ----------
    normx
    normxmin
    normxmax
    alpha

    Returns
    -------

    """
    normx = normx[(normx >= normxmin) * (normx <= normxmax)]
    if alpha == 1:
        return -1e-12, np.zeros(len(normx))
    C = 1 / np.sum(np.arange(normxmin, normxmax + 1) ** -alpha)
    dist = np.log(C) - alpha * np.log(normx)
    ll = np.sum(dist)

    return ll, dist


def exp_like(x, xmin, xmax, lam):
    """
    Exponential log likelihood. Matches with powerlaw library!

    Parameters
    ----------
    x
    xmin
    xmax
    lam

    Returns
    -------

    """
    if lam <= 0:
        return -1e-12, np.zeros(len(x))
    x = np.array(x)
    x = x[(x >= xmin) * (x <= xmax)]
    # Subtract minimum to turn it into a normalized PDF (Clauset et al 2009)
    x = x - xmin
    # MLE estimate for lambda says lambda_opt = 1/np.mean(x)
    # lam = 1/np.mean(x)
    N = len(x)
    dist = np.log(lam * np.exp(-lam * x))
    ll = N * np.log(lam) - N * lam * np.mean(x)

    return ll, dist


def tpl_like(x, xmin, xmax, alpha, lam):
    """
    Get the truncated power law likelihood function.
    Restricts values to be alpha > 1 and lambda > 0. Matches with powerlaw() library!

    Parameters
    ----------
    x
    xmin
    xmax
    alpha
    lam

    Returns
    -------

    """
    if alpha <= 1 or lam <= 0 or len(x) <= 5 or np.isnan(alpha) or np.isnan(lam):
        return -1e12, np.zeros(len(x))
    x = x[(x >= xmin) * (x <= xmax)]
    dist = (1 - alpha) * np.log(lam) - alpha * np.log(x) - lam * x - mylog(gammainc(1 - alpha, lam * xmin))
    ll = float(sum(dist))
    dist = myfloat(dist)

    # val = lam*xmin
    # dist = (1-alpha)*np.log(lam) - alpha*np.log(x) - lam*x - np.log(scipy.special.expn(alpha,val)*(val**(1-alpha)))
    # dist = (1-alpha)*np.log(lam) - alpha*np.log(x) - lam*x - np.log(scipy.special.expn(alpha,val)) - (1-alpha)*np.log(val)
    # ll = np.sum(dist)

    return ll, dist


def lognormal_like_truncated(data, xmin, xmax, mu, sigma):
    """
    The real likelihood function for the truncated lognormal distribution when known to be truncated between xmin and xmax.

    Parameters
    ----------
    data
    xmin
    xmax
    mu
    sigma

    Returns
    -------

    """
    data = data[(data >= xmin) * (data <= xmax)]
    myscale = np.exp(mu)
    dist = lognormlogpdf(data, s=sigma, scale=myscale) - ln(
        lognormcdf(xmax, s=sigma, scale=myscale) - lognormcdf(xmin, s=sigma, scale=myscale))
    ll = np.sum(dist)

    return ll, dist


def lognormal_like(data, xmin, xmax, mu, sigma):
    """
    The likelihood function used in Clauset et al 2009.
    Useful for lognormal data truncated from the left and only fitting for xmin < x < xmax and the underlying distribution is not truncated.

    Parameters
    ----------
    data
    xmin
    xmax
    mu
    sigma

    Returns
    -------

    """
    data = data[(data >= xmin) * (data <= xmax)]
    myscale = np.exp(mu)
    dist = lognormlogpdf(data, s=sigma, scale=myscale) - ln(1 - lognormcdf(xmin, s=sigma, scale=myscale))
    ll = np.sum(dist)

    return ll, dist


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MAXIMUM LIKELIHOOD (MLE) FITS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


@numba.njit
def find_pl(x, xmin, xmax=1e6, stepsize=None):
    """
    Find the power law between xmin and xmax, assuming continuous variable.
    v2 use minimize_scalar to be about 3-4x faster than nelder-mead.
    v3 prefer to use brent_findmin to be another 10x faster.

    Parameters
    ----------
    x : array
        The values to search for a power law within.
    xmin : float
        The minimum of the scaling regime.
    xmax : float
        The maximum of the scaling regime. The default is 1e6.
    stepsize : float, optional
        Added to ensure cross-compatability with discrete find_pl. Does not do anything in this function. The default is None.

    Returns
    -------
    alpha : float
        The optimal power law exponent.
    ll : float
        The log-likelihood at the optimal alpha.

    """
    xc = x[(x >= xmin) * (x <= xmax)]
    # mymean = lambda a: -pl_like(xc,xmin,xmax,a)[0]
    # myfit = optimize.minimize(mymean,2,method = 'Nelder-Mead', bounds = [(1,1e6)])
    # myfit = optimize.minimize_scalar(mymean, bounds = (1,30))
    alpha = brent_findmin(xc)
    ll = pl_like(xc, xmin, xmax, alpha)[0]

    return alpha, ll


@numba.njit
def find_pl_discrete(x, xmin, xmax, stepsize=None):
    """
    Find the power law in a discrete variable.

    Parameters
    ----------
    x : float array
        Contains the data to find the power law in. Should be discrete (i.e. only multiples of some dx value)
    xmin : float
        The minimum of the scaling regime.
    xmax : float
        The maximum of the scaling regime.
    stepsize : float, optional
        The stepsize of the discrete variable, i.e. the timestep for durations. If None, then stepsize will be set to min(x)/100 (i.e. small enough that the variable is assumed continuous). The default is None.

    Returns
    -------
    alpha : float
        The optimal power law exponent.
    ll : float
        The log-likelihood at the optimal alpha.

    """
    if stepsize is None:
        stepsize = min(x) / 100  # If not specified, assume the stepsize is small enough that we don't need to worry.

    normx = np.rint(x / stepsize)
    normxmin = np.rint(xmin / stepsize)
    normxmax = np.rint(xmax / stepsize)

    # Remove from consideration all events that are rounded down to zero, have xmin be equal to 1.
    if normxmin == 0:
        normxmin = 1
    if normxmax == 0:
        normxmax = 1

    normx = normx[(normx >= normxmin) * (normx <= normxmax)]
    alpha = brent_findmin_discrete(normx)
    ll = pl_like_discrete(normx, normxmin, normxmax, alpha)[0]

    return alpha, ll


def find_tpl(x, xmin, xmax=1e6):
    """
    Find the truncated power law parameters.
    Matches with the output from powerlaw().

    Parameters
    ----------
    x
    xmin
    xmax

    Returns
    -------

    """
    x = x[(x >= xmin) * (x <= xmax)]
    # initial_guess = [1 + len(x)/sum(np.log(x/xmin)), 1/np.mean(x)]
    initial_guess = [2, 1 / np.mean(x)]
    mymean = lambda par: -tpl_like(x, xmin, xmax, par[0], par[1])[0]
    opt_results = scipy.optimize.minimize(mymean, initial_guess, method='Nelder-Mead')
    alpha = opt_results.x[0]
    lam = opt_results.x[1]

    # Maximum likelihood
    ll = -opt_results.fun

    return alpha, lam, ll


def find_exp(x, xmin, xmax=1e6):
    """
    Find the best-matching exponential distribution using MLE.
    Matches with powerlaw().

    Parameters
    ----------
    x
    xmin
    xmax

    Returns
    -------

    """
    x = np.array(x)
    initial_guess = [1 / np.mean(x[(x >= xmin) * (x <= xmax)])]
    mymean = lambda par: -exp_like(x, xmin, xmax, par[0])[0]
    opt_results = scipy.optimize.minimize(mymean, initial_guess, method='Nelder-Mead')
    lam = opt_results.x[0]
    ll = -opt_results.fun

    return lam, ll


def momsolve_truncnorm(data, a, b):
    """
    Method of moments estimator of mu, sigma for truncated normal distribution.
    Broken as of Jul 5, 2024.

    Parameters
    ----------
    data
    a
    b

    Returns
    -------

    """
    n = len(data)

    def f(p):
        mu = p[0]
        sig = p[1]

        # PDFs
        phib = normpdf((b - mu) / sig)
        phia = normpdf((a - mu) / sig)

        # CDFs
        Phib = normcdf((b - mu) / sig)
        Phia = normcdf((a - mu) / sig)
        eq1 = mu - sig * ((phib - phia) / (Phib - Phia)) - np.mean(data)
        eq2 = mu * mu + sig * sig - sig * (((mu + b) * phib - (mu + a) * phia) / (Phib - Phia)) - np.mean(data * data)

        return eq1, eq2

    muinit, siginit = scipy.optimize.fsolve(f, [1, 1])

    return muinit, siginit


def find_trunc_norm(x, xmin=-np.inf, xmax=np.inf):
    """
    Find the truncated normal distribution.
    Likelihood function from simple probability distribution f(x) = phi(x)/(Phi(b)-Phi(a)) where phi(x) = pdf of N(0,1), and Phi(x) = cdf of N(0,1) at x

    Parameters
    ----------
    x
    xmin
    xmax

    Returns
    -------

    """
    x = x[(x >= xmin) * (x <= xmax)]
    n = len(x)

    # Negative likelihood function for truncated normal.
    def f(p):
        mu = p[0]
        sig = p[1]
        tot = n * np.log(normcdf(xmax, mu, sig) - normcdf(xmin, mu, sig)) - np.sum(normlogpdf(x, mu, sig))
        return tot

    inits = momsolve_truncnorm(x, xmin, xmax)
    outs = scipy.optimize.minimize(f, inits,
                                   method='Nelder-Mead')  # will try to guess mu between -20 and 20, and sigma between 0 and 20.
    mu, sigma = outs.x
    ll = -outs.val

    return mu, sigma, ll


def find_lognormal_from_normal(x, xmin=0, xmax=np.inf):
    """
    Fit the lognormal data by transforming it to normal, fitting using tnorm_mle, then reverse transforming.
    Gives same answer as fitting the lognormal log likelihood directly.

    Parameters
    ----------
    x
    xmin
    xmax

    Returns
    -------

    """
    # Convert the lognormal to a normal distribution and then fit that to get the MLE estimated parameters.
    anorm = np.log(xmin)
    bnorm = np.log(xmax)
    tnorm = np.log(x)

    mu, sigma = find_trunc_norm(tnorm, anorm, bnorm)[:-1]
    ll = lognormal_like_truncated(x, mu, sigma, xmin, xmax)

    return mu, sigma, ll


def find_lognormal_truncated(x, xmin=0, xmax=np.inf):
    """
    Find the MLE estimate of the mu and sigma of lognormal data, known to be truncated between xmin and xmax.

    Parameters
    ----------
    x
    xmin
    xmax

    Returns
    -------

    """
    x = x[(x >= xmin) * (x <= xmax)]

    # inits = [np.mean(ln(x)),np.std(ln(x))] #MLE estimate for untruncated lognormal
    xnorm = np.log(x)
    anorm = np.log(xmin)
    bnorm = np.log(xmax)
    # inits = momsolve_truncnorm(xnorm,anorm,bnorm) #use method of moments on truncated normal to estimate mu and sigma
    inits = [1, 1]

    # Negative likelihood function for truncated normal.
    fun = lambda p: -lognormal_like_truncated(x, xmin, xmax, p[0], p[1])[0]

    # Minimize the negative likelihood.
    # Testing suggests the value obtained from MLE is very close to the one obtained from method of moments, so bounds is not required.
    # Will try to guess mu between -20 and 20, and sigma between 0 and 20.
    outs = scipy.optimize.minimize(fun, inits, method='Nelder-Mead')
    mu, sigma = outs.x
    ll = -outs.fun  # Return to positive

    return mu, sigma, ll


def find_lognormal(x, xmin=0, xmax=np.inf):
    """
    Find the MLE estimate of the mu and sigma of lognormal data only left truncated from x > xmin and data are between xmin < x < xmax

    Parameters
    ----------
    x
    xmin
    xmax

    Returns
    -------

    """
    x = x[(x >= xmin) * (x <= xmax)]

    # inits = [np.mean(ln(x)),np.std(ln(x))] #MLE estimate for untruncated lognormal
    xnorm = np.log(x)
    anorm = np.log(xmin)
    bnorm = np.log(xmax)
    # inits = momsolve_truncnorm(xnorm,anorm,bnorm) #use method of moments on truncated normal to estimate mu and sigma
    inits = [np.mean(np.log(x)), np.std(np.log(x))]

    # Negative likelihood function for truncated normal.
    fun = lambda p: -lognormal_like(x, xmin, xmax, p[0], p[1])[0]

    # Minimize the negative likelihood. Testing suggests the value obtained from MLE is very close to the one obtained from method of moments, so bounds is not required.
    # Will try to guess mu between -20 and 20, and sigma between 0 and 20.
    outs = scipy.optimize.minimize(fun, inits, method='Nelder-Mead')
    mu, sigma = outs.x
    ll = -outs.fun  # Return to positive

    return mu, sigma, ll


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LOG-LIKELIHOOD RATIO (LLR) COMPARISONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def llr(dist1, dist2, nested=False):
    """
    The truncated power law is nested within the power law, so set "nested" to true when comparing those two.

    Parameters
    ----------
    dist1
    dist2
    nested

    Returns
    -------

    """
    length = len(dist1)
    llr = sum(dist1 - dist2)
    var = np.var(dist1 - dist2)
    if nested:
        p = 1 - scipy.stats.chi2.cdf(abs(2 * llr), 1)
    else:
        # Obtain p-value from distribution compare by using error function (assumes the test statistic is normally distributed with mean value llr and variance var)
        p = float(myerfc(abs(llr) / np.sqrt(2 * length * var)))

    return llr, p


def llr_wrap(x, xmin, xmax, totest=['power_law', 'exponential'], stepsize=None):
    """
    Wrapper function for llr.
    Finds the llr and p value for two distributions.
    Positive value and p <= p_thresh prefers the first function over the second.
    Negative number and p <= p_thresh prefers the second distribution.
    If p > p_thresh, then neither is significantly preferred between the two.
    Recommend p_thresh = 0.01 to reproduce results in Kepler manual analysis.

    Parameters
    ----------
    x: vector of (size/dur/vmax) data.
    xmin: the minimum of the scaling regime
    xmax: maximum of the scaling regime
    totest:
    stepsize:

    Returns
    -------

    """
    # Check if inputs for distributions are legal
    legal = ['power_law', 'exponential', 'lognormal', 'truncated_power_law']
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

    x = np.sort(x)

    # If the inputs are 'power_law' and 'truncated_power_law', then the inputs are nested versions of one another.
    # Note: according to Corrall et al 2019, the power law is nested within the truncated lognormal distribution.
    if (len(set(totest) - set(['power_law', 'truncated_power_law'])) == 0) or (
            len(set(totest) - set(['power_law', 'lognormal'])) == 0):
        nested = True
    else:
        nested = False

    # Get the right functions to compare
    findfuns = [None] * 2  # Functions to get the optimal values
    llrfuns = [None] * 2
    opts = [None] * 2
    dists = [None] * 2

    # Do the llr comparison against variable 'normx' which will hold the x values normalized against stepsize if stepsize is not none.
    normx = x
    normxmin = xmin
    normxmax = xmax
    if stepsize is not None:
        normx = np.rint(x / stepsize)
        normxmin = np.rint(xmin / stepsize)
        normxmax = np.rint(xmax / stepsize)

    checks = []
    for i in range(len(totest)):
        if totest[i] == 'power_law':
            # print('pl')
            findfuns[i] = find_pl
            llrfuns[i] = pl_like
            if stepsize is not None:
                findfuns[i] = lambda x, xmin, xmax: find_pl_discrete(x, xmin, xmax, stepsize=stepsize)
                llrfuns[i] = pl_like_discrete
        if totest[i] == 'truncated_power_law':
            # print('tpl')
            findfuns[i] = find_tpl
            llrfuns[i] = tpl_like
        if totest[i] == 'exponential':
            # print('exp')
            findfuns[i] = find_exp
            llrfuns[i] = exp_like
        if totest[i] == 'lognormal':
            # print('ln')
            findfuns[i] = find_lognormal_truncated
            llrfuns[i] = lognormal_like_truncated

        # In the case that we are testing against a discrete power law, always calculate the optimal value using find_pl_discrete on un-normalized data to avoid double-rounding error.
        opts[i] = findfuns[i](normx, normxmin, normxmax)[:-1]
        if totest[i] == 'power_law':
            # In the case that we are testing against a discrete power law, always calculate the optimal value using find_pl_discrete on un-normalized data to avoid double-rounding error.
            opts[i] = findfuns[i](x, xmin, xmax)[:-1]
        dists[i] = llrfuns[i](normx, normxmin, normxmax, *opts[i])[-1]
        checks = checks + list(opts[i])

    checks = arr(checks)
    # print(dists)
    # Now that the dist lists are populated, get the optimal values using MLE
    if np.any(np.isnan(checks)):
        return -1e10, -1

    ll, p = llr(dists[0], dists[1], nested=nested)

    return ll, p


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ANDERSON-DARLING (AD) TEST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def ad(s, d, smin, smax, dmin, dmax):
    """

    Parameters
    ----------
    s
    d
    smin
    smax
    dmin
    dmax

    Returns
    -------

    """
    sc = s[(s >= smin) * (s <= smax)]
    dc = d[(d >= dmin) * (d <= dmax)]

    # Get events larger than smin to compare for truncated power law
    scm = s[(s >= smin)]
    dcm = d[(d >= dmin)]

    tau_pl = find_pl(sc, smin, smax)
    alpha_pl = find_pl(dc, dmin, dmax)

    # Truncated power law (no xmax!)
    tau_tpl = find_tpl(scm, smin)
    alpha_tpl = find_tpl(dcm, dmin)

    # Lognormal
    sizelog = find_lognormal(sc, smin, smax)
    durlog = find_lognormal(dc, dmin, dmax)

    # Exponential
    sizeexp = find_exp(sc, smin)
    durexp = find_exp(dc, dmin)

    # Simulated datasets
    spl_sim = pl_gen(1000, smin, smax, tau_pl[0])
    dpl_sim = pl_gen(1000, dmin, dmax, alpha_pl[0])
    stpl_sim = tpl_gen(1000, smin, tau_tpl[0], tau_tpl[1])
    dtpl_sim = tpl_gen(1000, dmin, alpha_tpl[0], alpha_tpl[1])
    slog_sim = lognormal_gen(1000, smin, smax, sizelog[0], sizelog[1])
    dlog_sim = lognormal_gen(1000, dmin, dmax, durlog[0], durlog[1])
    sizeexp_sim = exp_gen(1000, smin, sizeexp[0])
    durexp_sim = exp_gen(1000, dmin, durexp[0])

    # Do AD tests
    ad_spl = scipy.stats.anderson_ksamp([sc, spl_sim])
    ad_dpl = scipy.stats.anderson_ksamp([dc, dpl_sim])
    ad_stpl = scipy.stats.anderson_ksamp([scm, stpl_sim])
    ad_dtpl = scipy.stats.anderson_ksamp([dcm, dtpl_sim])
    ad_slog = scipy.stats.anderson_ksamp([sc, slog_sim])
    ad_dlog = scipy.stats.anderson_ksamp([dc, dlog_sim])
    ad_sizeexp = scipy.stats.anderson_ksamp([sc, sizeexp_sim])
    ad_durexp = scipy.stats.anderson_ksamp([dc, durexp_sim])

    return ad_spl, ad_stpl, ad_slog, ad_sizeexp, ad_dpl, ad_dtpl, ad_dlog, ad_durexp
