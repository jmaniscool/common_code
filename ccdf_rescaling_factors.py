"""
Created Sep 07 2024 (Ethan Mullen)
Updated Nov 14 2024 (Ethan Mullen)
"""
import scipy.stats as stats
import numpy as np


def ccdf_rescaling_factors(data=None, nth_moment=None, pdf_exp=None, moment_center=0):
    """
    Returns the rescaling factors for x- and y-axes of CCDF.
    Should be implemented by dividing the x- and y-arrays by their corresponding rescaling factors.

    Parameters
    ----------
    data: (Array or list, required)
    nth_moment: (Integer, required)
        The order of the moment
    pdf_exp: (Float, required)
    moment_center: (Float, optional)
        By default, scipy calculates "centered" moments.
        Default value of this parameter ensures that moments are simply the sum of the values raised to the nth power divided by the number of values.

    Returns
    -------
    [0] x-axis rescaling factor
    [1] y-axis rescaling factor
    """
    # This is just so that Python stops getting mad at me for setting the default value to None.
    if pdf_exp is None:
        pdf_exp = 1

    if data is None or nth_moment is None or pdf_exp is None:
        print('Please input data, moment order, and PDF exponent. See docstring for help.')
        return np.array([]), np.array([])

    data_moment = stats.moment(data, moment=nth_moment, nan_policy='omit', center=moment_center)

    x_rescaling_factor = data_moment ** (1 / ((nth_moment + 1) - pdf_exp))
    y_rescaling_factor = data_moment ** ((1 - pdf_exp) / ((nth_moment + 1) - pdf_exp))

    return x_rescaling_factor, y_rescaling_factor
