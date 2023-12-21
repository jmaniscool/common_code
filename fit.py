# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:22:00 2020
Updated on April 18, 2021
Updated again for speed on December 21, 2023.

Create a Python minimum least-squares fitting function using python lambda functions.

Variables:
    xdata: a list or np.array of data x-values
    ydata: a list or np.array of data y-values
    func: a lambda function in the following format (example is y = ax^2 + bx + c):
        func = lambda x,p0,p1,p2: p0 + p1*x + p2*x^2. Make sure x is the leading variable, and only one number is assigned to each parameter!

    xmin: minimum x value to consider.
    ymin: minimum y value to consider.
    ci: the confidence interval to report errors for, where
    test: either 'Rsq' or 'chi2'. Rsq for R-squared test (good for linear models), and chi2 for chi-squared test.
        IMPORTANT NOTE: According to some people, a high p-value on the chi2 test is evidence that the data is described by the model,
        though Jordan believes that this is a misrepresentation of the concept of p values. The purpose of any
        statistical test is to test against a null hypothesis H0 with an alternative hypothesis H1. If
        p < pcrit (usually 0.05), then we can reject the null hypothesis and accept the alternative hypothesis.
        For a chi squared test, the alternative hypothesis is that the two data are drawn different underlying distributions.
        However, a high p-value is not evidence that the null is true.
        
        Thus, Jordan thinks the way to interpret the chi squared test is as follows:
            --if p < pcrit, the data are not drawn from the distribution given by the model.
            --if p > pcrit, the data MAY be drawn from the same distribution as the underlying model. A higher p value is not better evidence for this claim.
        
@author: Jordan
"""
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from scipy import stats

def fit(xdata,ydata,func = lambda x,p0,p1: p0*x+p1, xmin = None, xmax = None, ci = 0.95, test = 'rsq', print_chi2 = True):

    if xmin == None:
        xmin = min(xdata)
    if xmax == None:
        xmax = max(xdata)
        
    if (xmin > xmax) or (xmax < xmin):
        print("Error! Make sure xmin < xmax.")
        return -1,-1,-1
        
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    
    filt = ~np.isnan(ydata)*~np.isnan(xdata)*(xdata > xmin)*(xdata < xmax)
    
    x = xdata[filt]
    y = ydata[filt]    
    
    z = stats.norm.ppf(1-ci/2) #get z score
    
        
    #old filter
    #get the fit areas. Convert to float to ensure correct typing
    #xnan = np.isnan(xdata)
    #ynan = np.isnan(ydata)
    #x = np.array([xdata[i] for i in range(len(xdata)) if (xdata[i] >= minval) & (xdata[i] <= maxval) & (xnan[i] == False) & (ynan[i] == False)])
    #y = np.array([ydata[i] for i in range(len(xdata)) if (xdata[i] >= minval) & (xdata[i] <= maxval) & (ynan[i] == False) & (xnan[i] == False)])
    
    
    #use * operator to unpack xp into list of inputs for function, starting with x
    p1, pcov = optimize.curve_fit(func,x,y)
    err = z*np.diag(pcov)**0.5 #standard error on parameters is sqrt of diagnoal elements. Multiply by z-score to get CIs assuming normally distributed parameters.

    #use * operator to unpack xp into a tuple for insertion into the function    
    yguess = func(x,*p1)
    
    #calculate the R^2 of the fit
    if test == 'rsq' or 'Rsq' or 'r_squared' or 'R_squared' or 'r2' or 'R2':
        ybar = np.mean(y)
        sst = sum((y-ybar)**2)    
        ssres=sum((yguess-y)**2)    
        rsq = 1 - ssres/sst
        return p1,err,rsq
    
    elif test == 'chi2' or 'Chi2' or 'chi_squared':
        chi_squared = np.sum(((y-yguess) / yguess) ** 2)
        
        pval = 1- stats.chi2.cdf(chi_squared,len(y)-len(p1))
        if pval <= 1-ci:
            message = "Chi squared test suggests model does not fit data with p = %.2e less than pcrit = %.2e" % (pval, 1-ci)
        else:
            message = "Chi squared test cannot reject null hypothesis that data are drawn from the same distribution with p = %.2e greater than pcrit = %.2e" % (pval, 1-ci)
        
        if print_chi2 == True:
            print(message)
        
        return p1,err,pval
    else:
        print("Error: Please input either chi2 or rsq for the test. Returning.")
        return -1,-1,-1