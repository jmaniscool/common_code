# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:27:58 2024

@author: sickl

Hold the brent root finder algorithm used for finding the power law of data vector x, where xmin <= x <= xmax
"""
import numba
import numpy as np

ln = np.log

#find the alpha using Brent's algorithm. This version is about 10x faster than the previous approach and gives the same answer, thanks to numba
#Gives same results as Scipy version which uses a c backend as implemented at https://github.com/scipy/scipy/blob/v1.13.1/scipy/optimize/Zeros/brentq.c
#Fortran version that's slightly different is implemented in https://websites.pmc.ucsc.edu/~fnimmo/eart290c_17/NumericalRecipesinF77.pdf.
@numba.njit
def brent_findmin(x,blo = 1.0, bhi = 20.0, xtol = 1e-12, rtol = 8.881784197001252e-16, maxiter = 100):
    ln = np.log
    x = np.sort(x)
    xmin = x[0]
    xmax = x[-1]
    n = len(x)
    if np.sum(x) == x[0]*n:
        return np.nan
    S = np.sum(np.log(x))
    def f(alpha):
        #added edge case for alpha near 1
        #test for alpha = 1
        if alpha == 1:
            val = -ln(ln(xmax/xmin)) - S/n #equation from Deluca & Corrall 2013, equation 12.
            if val < 0: #sometimes the value of this function will be less than zero (i.e. for lognormally distributed data). In that case, just return a positive value because it's an error.
                return 100
            return val
        #large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
        test_xmin = np.log10(xmin)*(-alpha+1)
        if test_xmin > 100:
            return -10
        
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
    
    #hold previous, current, and blk (?) values
    xpre = blo #previous estimate of the root
    xcur = bhi #current estimate of the root
    xblk = np.nan #holds value of x
    fpre = f(xpre)
    fcur = f(xcur)
    fblk = np.nan #hold value of f(x) (?)

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

def discrete(alpha,x):
    xmin = np.min(x)
    xmax = np.max(x)
    vals = np.arange(xmin,xmax+1)
    n = len(x)
    #added edge case for alpha near 1
    #test for alpha = 1
    if alpha == 1:
        val = -ln(ln(xmax/xmin)) - np.sum(np.log(x))/n #equation from Deluca & Corrall 2013, equation 12.
        if val < 0: #sometimes the value of this function will be less than zero (i.e. for lognormally distributed data). In that case, just return a positive value because it's an error.
            return 100
        return val
    #large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
    test_xmin = np.log10(xmin)*(-alpha+1)
    if test_xmin > 100:
        return -10
    
    y = - np.sum(ln(x)) + n*(np.sum(ln(vals) * vals**(-alpha)))/(np.sum(vals**-alpha))
    
    return y

#discrete find_pl for brent_findmin. Assumes that xmin <= x <= xmax and x are integers.
#The continuous function to find a root over. Validated.
@numba.njit
def brent_findmin_discrete(x,blo = 1.0, bhi = 20.0, xtol = 1e-12, rtol = 8.881784197001252e-16, maxiter = 100):
    ln = np.log
    x = np.sort(x)
    xmin = x[0]
    xmax = x[-1]
    vals = np.arange(xmin,xmax+1)
    n = len(x)
    if np.sum(x) == x[0]*n:
        return np.nan
    logvals = ln(vals)
    S = np.sum(ln(x))
    def f(alpha):
        #added edge case for alpha near 1
        #test for alpha = 1
        if alpha == 1:
            val = -ln(ln(xmax/xmin)) - np.sum(np.log(x))/n #equation from Deluca & Corrall 2013, equation 12.
            if val < 0: #sometimes the value of this function will be less than zero (i.e. for lognormally distributed data). In that case, just return a positive value because it's an error.
                return 100
            return val
        #large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
        test_xmin = np.log10(xmin)*(-alpha+1)
        if test_xmin > 100:
            return -10
        
        #Add in approximation for low alpha if necessary.
        """
        #if the tested alpha is very low, use a taylor approximation
        if alpha < 1 + 1e-7:
            y = alpha-1
            beta = y*ln(xmax/xmin)
            gam = ln(xmax/xmin) - y*ln(xmax)*ln(xmax) + y*ln(xmin)*ln(xmin)
        else:
            beta = -xmax**(-alpha+1) + xmin**(-alpha+1)
            gam = xmax**(-alpha+1)*ln(xmax) - xmin**(-alpha+1)*ln(xmin)
            
        y = n/(alpha - 1) - S - n*(gam/beta)
        """
        y = - S + n*(np.sum(logvals * vals**(-alpha)))/(np.sum(vals**-alpha))
        
        return y
    
    #hold previous, current, and blk (?) values
    xpre = blo #previous estimate of the root
    xcur = bhi #current estimate of the root
    xblk = np.nan #holds value of x
    fpre = f(xpre)
    fcur = f(xcur)
    fblk = np.nan #hold value of f(x) (?)

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

from numba import cuda
import math
@cuda.jit
def brent_findmin_cuda(x, out):
    blo = 1.0
    bhi = 20.0
    xtol = 1e-12
    rtol = 8.881784197001252e-16
    maxiter = 100
    ln = math.log
    log10 = math.log10
    xmin = x[0]
    xmax = x[-1]
    n = len(x)
    sumx = 0
    S = 0
    out[0] = -1 #set the default value of out
    for i in range(len(x)):
        sumx = sumx + x[i]
        S = S + ln(x[i])
    if sumx == x[0]*n:
        return
    def f(alpha):
        #added edge case for alpha near 1
        #test for alpha = 1
        if alpha == 1:
            val = -ln(ln(xmax/xmin)) - S/n #equation from Deluca & Corrall 2013, equation 12.
            if val < 0: #sometimes the value of this function will be less than zero (i.e. for lognormally distributed data). In that case, just return a positive value because it's an error.
                return 100
            return val
        #large values of test_xmin lead to undefined behavior due to float imprecision, limit approaches -inf. with derivative +inf
        test_xmin = log10(xmin)*(-alpha+1)
        if test_xmin > 100:
            return -10
        
        #if the tested alpha is very low, use a taylor approximation
        if alpha < 1 + 1e-7:
            y = alpha-1
            beta = y*ln(xmax/xmin)
            gam = ln(xmax/xmin) - y*ln(xmax)*ln(xmax) + y*ln(xmin)*ln(xmin)
        else:
            beta = -pow(xmax,-alpha+1) + pow(xmin,-alpha+1)
            gam = pow(xmax,-alpha+1)*ln(xmax) - pow(xmin,-alpha+1)*ln(xmin)
            
        y = n/(alpha - 1) - S - n*(gam/beta)
        
        return y
    
    #hold previous, current, and blk (?) values
    xpre = blo #previous estimate of the root
    xcur = bhi #current estimate of the root
    xblk = -1 #holds value of x
    fpre = f(xpre)
    fcur = f(xcur)
    fblk = -1 #hold value of f(x) (?)

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
        out[0] = fpre
        return
    if fcur == 0:
        out[0] = fcur
        return
    tol1 = -1
    
    #main loop
    for i in range(maxiter):
        #if fpre and fcur are both not zero and fpre has a different sign from fcur
        if ((fpre != 0)*(fcur != 0)*(fpre/abs(fpre) != fcur/abs(fcur))):
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
            out[0] = xcur
            return
        
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
            xcur = xcur + (sbis)/abs(sbis)*delta
        
        fcur = f(xcur) #another function call
    
    #print('Max iters achieved.')
    out[0] = xcur
    return