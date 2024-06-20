"""
Created Aug 30 2023
Updated Jun 19 2024
Author: Ethan

Contains one function that generates a power law line
"""

import numpy as np


def linemaker(slope, intercept, xmin, xmax, ppd=40):
    """
    Returns X and Y arrays of a power law line over a specified region with a specified point density.
    :param slope: Power law slope; INT or FLOAT.
    :param intercept: Intercept of the line; list of INT or FLOAT.
    :param xmin: Minimum x-value the line will appear over; INT or FLOAT.
    :param xmax: Maximum x-value the line will appear over; INT or FLOAT
    :param ppd: Number of log-spaced points per decade to evaluate the line at; INT.
    :return:
    [0] x_vals: X values of the line; Numpy array.
    [1] y_vals: Y values of the line; Numpy array.
    """

    if not (isinstance(slope, int) or isinstance(slope, float)) or slope == 0:
        print('Please enter the slope as a nonzero int or float')
        return 'temp'
    if not(isinstance(intercept, list)):
        print('Please enter an intercept of the line as a list [x, y] of ints or floats')
        return 'temp'
    for coord in intercept:
        if not (isinstance(coord, int) or isinstance(coord, float)):
            print('Please enter an intercept of the line as a list [x, y] of ints or floats')
            return 'temp'
    if not (isinstance(xmin, int) or isinstance(xmin, float)):
        print('Please enter an xmin as an int or float')
        return 'temp'
    if not (isinstance(xmax, int) or isinstance(xmax, float)):
        print('Please enter an xmax as an int or float')
        return 'temp'
    if (not (isinstance(ppd, int) or isinstance(ppd, float))) or ppd <= 0:
        print('Please enter the desired points-per-decade as a positive int')
        return 'temp'

    # Take the log of all the inputs
    log_x_intercept, log_y_intercept = np.log10(intercept[0]), np.log10(intercept[1])
    log_xmin, log_xmax = np.log10(xmin), np.log10(xmax)

    # Calculate the y-intercept of the line on log axes
    log_b = log_y_intercept - slope * log_x_intercept

    # Get the x- and y-values of the line as arrays
    x_vals = np.logspace(log_xmin, log_xmax, round(ppd * (log_xmax - log_xmin)))
    y_vals = (10 ** log_b) * (x_vals ** slope)

    return x_vals, y_vals
