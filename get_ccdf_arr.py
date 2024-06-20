"""
Original CCDF code: Alan Long Jun 10 2016
Vectorized & edited by Jordan on Dec 19 2023
Edited by Ethan on Jun 12 2024
Edited slightly by Jordan on June 20 2024 (make the inputs to the CCDF function required)

This code takes data as a list or numpy array and returns its complementary cumulative distribution function (CCDF).
- Returns two lists: histx and histy, the x and y values for the ccdf.
- If input lists have negative, NaN, or inf values, it will throw them out and proceed normally.
- Has two separate functions:
[1] ccdf(): This is the standard method. Takes in the full list of data.
[2] ccdf2(): This method allows you to send in the unique values of your data and their counts. Useful if you have a
             very large data array that is best kept as unique values & their counts. These do not need to be sorted,
             but they must have the same length and each data value must have its corresponding count at the same index.
- There are two inputs methods for both functions:
[1] 'scipy': CCDF is defined as P(X > x). The data is appended with a zero at the beginning such that
             P(X > 0) = 1 is the first pair in histx & histy.
[2] 'dahmen': CCDF is defined as P(X >= x). The data is unchanged, so the first pair in histx & histy is
              P(X >= [smallest array value]) = 1.
"""

import numpy as np


#Jordan's version, depreciated as of 6-20-24.
#Simpler and faster version of get_cum_dist() function.
#Currently between 50% and 200%+ speed improvement over get_cum_dist().
#Can be made faster if unique() is replaced by a jit-able function.
#Option to freely switch between two definitions of CCDF: option "dahmen" is C(x) = Pr(x >= X) (inclusive), while option "scipy" is C(x) = Pr(x > X) (exclusive)
def get_ccdf_arr(data, method = 'scipy'):
    
    data = np.array(data) #convert to array
    if len(data) == 0:
        return np.array([]),np.array([])
    
    data = data[(data > 0)*~np.isnan(data)*~np.isinf(data)]
    
    histx = np.sort(data)
    N = len(histx)
    histy = np.zeros(N)
    
    #get the unique values greater than zero, not nans and not infs
    _, counts = np.unique(histx, return_counts = True)
    
    #get cumulative counts for the unique points
    ccounts = np.cumsum(counts)
    
    indices = np.zeros(len(counts) + 1)
    indices[0] = 0
    indices[1:] = ccounts
    
    
    #Dahmen group's method. The CCDF is defined as Pr(x >= X) (includes current value)
    if method == 'dahmen':
        
        #start counting at zero
        ccounts[1:] = ccounts[0:-1]
        ccounts[0] = 0
        
        ccounts = 1 - ccounts/N
    
    k = 0
    for i in range(len(histy)):
        k += i == indices[k+1]
        histy[i] = ccounts[k]
        
    
    #if method is scipy, then use definition of CCDF as Pr(x > X) (traditional method)
    if method == 'scipy':            
        histy = 1 - histy/histy[-1]

    
    return histx,histy





#Ethan's CCDF function, much faster and better for large data. Should be preferred for floats
def ccdf(data, method='scipy'):
    """
    :param data: Input data as a list or numpy array.
    :param method: (String) Choice between representing CCDF as P(X > x) ('scipy') or P(X >= x) ('dahmen').
    :return:
    [0] histx = X-values in CCDF
    [1] histy = Y-values in CCDF
    """

    data = np.array(data)
    if len(data) == 0:
        print('Data array is empty.')
        return np.array([]), np.array([])

    if method != 'scipy' and method != 'dahmen':
        print('Please choose between two methods: \'scipy\' or \'dahmen\'.')
        return np.array([]), np.array([])

    # Take only positive values, non-NaNs, and non-Infs
    data = data[(data > 0) * ~np.isnan(data) * ~np.isinf(data)]

    # Get the unique values and their counts
    vals, counts = np.unique(data, return_counts=True)
    # Sort both the values and their counts the same way
    histx = vals[np.argsort(vals)]
    counts = counts[np.argsort(vals)]

    # P(X > x)
    if method == 'scipy':
        histx = np.insert(histx, 0, 0)

        # Get cumulative counts for the unique points
        cum_counts = np.cumsum(counts)

        # Get the total number of events
        total_count = cum_counts[-1]

        # Start constructing histy by saying that 100% of the data should be greater than 0
        histy = np.ones(len(counts) + 1)
        histy[1:] = 1 - (cum_counts / total_count)

    # P(X >= x)
    elif method == 'dahmen':
        cum_counts = np.cumsum(counts)
        # Now we insert a 0 at the beginning of cum_counts.
        # Since Pr(X >= x) = 1 - Pr(X < x), we can get the second term from this newly expanded cum_counts
        cum_counts = np.insert(cum_counts, 0, 0)

        total_counts = cum_counts[-1]

        histy = (1 - (cum_counts / total_counts))[:-1]

    return histx, histy


#Get the CCDF when the valuse and data_counts are known. Best when data are ints, not floats.
def ccdf_unique(data_vals, data_counts, method='scipy'):
    """
    :param data_vals: Unique values of input as a list or numpy array.
    :param data_counts: Counts corresponding to data_vals.
    :param method: (String) Choice between representing CCDF as P(X > x) ('scipy') or P(X >= x) ('dahmen').
    :return:
    [0] histx = X-values in CCDF
    [1] histy = Y-values in CCDF
    """

    data_vals = np.array(data_vals)
    data_counts = np.array(data_counts)
    if len(data_vals) == 0 or len(data_counts) == 0:
        print('Either the data array or the count array is empty.')
        return np.array([]), np.array([])

    if len(data_vals) != len(data_counts):
        print('The lengths of the data array and the count array do not match.')
        return np.array([]), np.array([])

    if method != 'scipy' and method != 'dahmen':
        print('Please choose between two methods: \'scipy\' or \'dahmen\'.')
        return np.array([]), np.array([])

    histx = data_vals[np.argsort(data_vals)]
    counts = data_counts[np.argsort(data_vals)]

    if method == 'scipy':
        histx = np.insert(histx, 0, 0)
        cum_counts = np.cumsum(counts)
        total_count = cum_counts[-1]
        histy = np.ones(len(counts) + 1)
        histy[1:] = 1 - (cum_counts / total_count)

    elif method == 'dahmen':
        cum_counts = np.cumsum(counts)
        cum_counts = np.insert(cum_counts, 0, 0)
        total_counts = cum_counts[-1]
        histy = (1 - (cum_counts / total_counts))[:-1]

    return histx, histy
