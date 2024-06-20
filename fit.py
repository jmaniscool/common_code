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
    ci: the confidence interval to report errors for, where ci = 0.95 represents the 95% confidence interval
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
    
    sigma: a Nx1 vector of standard deviations. Will be used to weight the fit. If none are given, just use 1 for everything.
        
@author: Jordan

Bug in confidence interval calculation fixed by Ethan
"""
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from scipy import stats
import sklearn
import sklearn.metrics

def fit(xdata,ydata,func = lambda x,p0,p1: p0*x+p1, xmin = None, xmax = None, sigma = None, ci = 0.95, test = 'rsq', print_chi2 = True):

    if xmin == None:
        xmin = min(xdata)
    if xmax == None:
        xmax = max(xdata)
    
    sigma_given = True
        
    if sigma is None:
        sigma_given = False
        sigma = np.ones(len(xdata))
        
    if (xmin > xmax) or (xmax < xmin):
        print("Error! Make sure xmin < xmax.")
        return -1,-1,-1
        
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    
    filt = ~np.isnan(ydata)*~np.isnan(xdata)*(xdata > xmin)*(xdata < xmax)
    
    x = xdata[filt]
    y = ydata[filt]    
    sigma = sigma[filt]
    
    z = stats.norm.ppf((1+ci)/2) #get z score (should be 1.96 for 95% CI)
    
        
    #old filter
    #get the fit areas. Convert to float to ensure correct typing
    #xnan = np.isnan(xdata)
    #ynan = np.isnan(ydata)
    #x = np.array([xdata[i] for i in range(len(xdata)) if (xdata[i] >= minval) & (xdata[i] <= maxval) & (xnan[i] == False) & (ynan[i] == False)])
    #y = np.array([ydata[i] for i in range(len(xdata)) if (xdata[i] >= minval) & (xdata[i] <= maxval) & (ynan[i] == False) & (xnan[i] == False)])
    
    
    #use * operator to unpack xp into list of inputs for function, starting with x
    if sigma_given == True:
        p1, pcov = optimize.curve_fit(func,x,y, sigma = sigma, absolute_sigma = True)
    else:
        p1, pcov = optimize.curve_fit(func,x,y)
    err = z*np.diag(pcov)**0.5 #standard error on parameters is sqrt of diagnoal elements. Multiply by z-score to get CIs assuming normally distributed parameters.

    #use * operator to unpack xp into a tuple for insertion into the function    
    yguess = func(x,*p1)
    
    
    #calculate the R^2 of the fit.
    weights = 1/(sigma**2) #using standard weighting 1/Var(x)
    
    if test == 'rsq' or test == 'Rsq' or test == 'r_squared' or test == 'R_squared' or test == 'r2' or test == 'R2' or test == 'adjusted_rsq' or test == 'adj_rsq' or test == 'adj_Rsq':
        
        
        
        #something is wrong with this but I can't figure it out right now :(
        """
        ybar = np.average(y, weights = weights)
        sst = np.sum(weights*(y-ybar)**2, dtype = np.float64)
        ssres = np.sum(weights*(y-yguess)**2)
        rsq = 1 - (ssres/sst)
        """
        
        #instead, use sklearn since it's well-documented.
        rsq = sklearn.metrics.r2_score(y,yguess, sample_weight = weights)
        
        #if adjusted, adjust using the normal definition.
        if test == 'adjusted_rsq' or test == 'adj_rsq' or test == 'adj_Rsq':
            k = len(p1)
            n = len(y)
            rsq = 1 - (1-rsq)*((n-1)/(n-k-1))
        
        
        return p1,err,rsq
    
    elif test == 'chi2' or test == 'Chi2' or test == 'chi_squared':
        #note that chi2 is usually not a good choice for interpreting nonlinear fits. This is because a scaling factor on y can change the reported chi2.
        #see https://arxiv.org/pdf/1012.3754.pdf
        chi_squared = np.sum((y-yguess) ** 2)
        
        #TESTING: currently trying by testing the hypothesis that y-yguess should be normally distributed?
        #chi_squared = np.sum((y-yguess)**2 / (np.var(y-yguess)))
        #print(chi_squared)
        
        pval = 1- stats.chi2.cdf(chi_squared,len(y)-len(p1))
        
        
        if pval <= 1-ci:
            message = "Chi squared test suggests model does not fit data with p = %.2e less than pcrit = %.2e" % (pval, 1-ci)
        else:
            message = "Chi squared test cannot reject null hypothesis that data are drawn from the same distribution with p = %.2e greater than pcrit = %.2e" % (pval, 1-ci)
        
        if print_chi2 == True:
            print(message)
            
        return p1, err, pval
            
    #note that chi2 is usually not a good choice for interpreting nonlinear fits. This is because a scaling factor on y can change the reported chi2.
    #see https://arxiv.org/pdf/1012.3754.pdf
    elif test == 'adjchi2' or test == 'adj_chi2' or test == 'adjusted_chi2' or test == 'adjusted chi2':
        adjusted_chi_squared = np.sum((y-yguess)**2 / ((len(y)-len(p1)))) #using adjusted chi squared. Should be not too different from 1 for a reasonable fit. But, chi squared can be easily messed up by a scaling factor.
        
        #TESTING: currently trying by testing the hypothesis that y-yguess should be normally distributed?
        #adjusted_chi_squared = np.sum((y-yguess)**2 / (np.var(y-yguess)*(len(y)-len(p1))))
        
        return p1,err,adjusted_chi_squared
    else:
        print("Error: Please input either chi2 or rsq for the test. Returning.")
        return -1,-1,-1