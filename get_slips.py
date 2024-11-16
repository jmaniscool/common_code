"""
Created Jun 14 2016 (Alan Long)
Updated Oct 21 2023 (Jordan Sickle)
Updated Nov 16 2024 (Ethan Mullen)

-- Past versions by Tyler Salners, Alan Long, and Jim Antonaglia
-- Jordan edits:
    -- v2: Edited to get rid of ragged array output.
    -- v3: Edited to deal with very low time resolution appropriately.
    -- v4: Edited to appropriately consider start/end of avalanches when considering size.
        -- Also, use sliding median to get time varying threshold, and median absolute difference to get the standard deviation of noise.

-- This code will search for avalanches as points in the 'velocity' curve which exceed a threshold.
-- The size is the amount of displacement (decrease if drops == True) that occurs while the velocity is above a threshold.
-- If displacement is given, the code will do a numerical derivative using np.diff to find the velocity.
-- If velocity is given, the code will do a numerical integral using np.cumsum to find the displacement.
    -- USE get_slips_vel IF YOU HAVE A VELOCITY SIGNAL AND YOU WANT TO GUARANTEE NO NEGATIVE SIZES
-- If time vector is not given, the code will assume a time vector composed of index numbers (i.e. the avalanche velocity is in terms of displacement per timestep).
-- One of either displacement/velocity is required to be given, or both.
"""
import numpy as np
from scipy import integrate
import scipy
from scipy import ndimage
from scipy import stats


def get_slips_wrap(displacement=None, velocity=None, time=None, drops=True, threshold=0, mindrop=0, threshtype='median', window_size=101):
    """
    Wrapper function for the vectorized and fixed version of get_slips.
    Extracts basic avalanche statistics from provided data.
    Please ready parameter descriptions carefully!

    Parameters
    ----------
    displacement: (List or array-like; REQUIRED)
        Time series data to be analyzed for avalanches IF data is some accumulated quantity.
        I.e., net displacement or accumulated stress over time.
        An avalanche in this perspective is a "SLIP", in the parlance of the Dahmen Group.
    velocity: (List or array-like; REQUIRED)
        Time series data to be analyzed for avalanches IF data is some quantity where at each time a new value is acquired.
        I.e., number of spins flipped in one timestep of the random-field Ising model (RFIM) or the number of cell failures in one timestep in a slip model.
        An avalanche in this perspective is a "slip RATE", in the parlance of the Dahmen Group.
    time: (List or array-like; REQUIRED)
        Time vector in data units.
        Defaults to an index array, i.e., an array ranging from 0 to N - 1, where N is the length of the input data.
    drops: (Boolean; OPTIONAL)
        Default value is TRUE.
        Whether to scan the time series for drops in the data.
    threshold: (Float; OPTIONAL)
        Default value is 0.
        Number of standard deviations above the average velocity a fluctuation must be before it is considered an event.
        Recommend 0 for most applications.
        Setting this equal to -1 forces a zero-velocity threshold on velocity curve.
            This is useful for simulations since there's little to no noise to be mistaken for an avalanche.
    mindrop: (Float; OPTIONAL)
        Default value is 0.
        Minimum size required to be counted as an event.
        Recommend 0 when starting analysis, but should be set higher when i.e. data culling to find the true value of tau.
    threshtype: (String; OPTIONAL)
        Default value is 'median'.
        What type of threshold to use. Options:
        'median'
            -- Uses the median velocity instead of the mean velocity to quantify the average velocity.
            -- Works best in signals with many excursions (i.e., many avalanches) and is not sensitive to outliers.
            -- Threshold is calculated using median absolute deviation (MAD) with this option instead of standard deviation because it more accurately describes the dispersion of the noise fluctuations while ignoring the avalanches.
        'mean'
            -- This is the traditional method.
            -- The threshold is compared to the mean velocity, (displacement[end]-displacement[start])/(time[end]-time[start]).
            -- Threshold is calculated using the standard deviation with this setting.
            -- This method has some major issues in non-simulation environments, as the standard deviation is very sensitive to large excursions from the noise floor
        'sliding_median'
            -- Uses a sliding median of width window_length to obtain a sliding median estimate of the background velocity.
            -- Useful when the background rate of the process is not constant over the course of the experiment.
            -- Threshold is calculated using median absolute deviation with this option and follows the change in average velocity while accurately describing the dispersion of the noise.
    window_size: (Int; OPTIONAL)
        Default value is 1.
        The window size, in datapoints, to be used when calculating the sliding median.
        Should be set to be much longer than the length of an avalanche in your data.
        Jordan found setting window_size = 3% the length of the data (in one case) was good, but this value can be anywhere from just a few hundred datapoints (very short avalanches, many datapoints) to up to 10-20% of the total length of the signal (when the data are shorter but contain several very long avalanches).
        This is worth playing with!

    Returns
    -------
    [0] velocity: list of lists
        -- A list of avalanche velocity curves.
        -- E.g. v[8] will be the velocity curve for the 9th avalanche.
        -- Velocity curve will always begin and end with 0s because the velocity had to cross 0 both times in its run.
    [1] times: list of lists
        -- A list of avalanche time curves.
        -- E.g. t[8] will be the corresponding times at which v[8] velocity curve occured.
        -- The time curve will have the 0s in the velocity curve occur at t_start - ts/2 and t_end + ts/2 as a 0-order estimate of when the curve intersected with zero.
    [2] sizes: list of floats
        -- List of avalanche sizes.
        -- The size is defined as the amount the displacement changes while the velocity is strictly above the threshold.
        -- Each size is corrected by the background rate.
        -- That is, (background rate)*(duration) is removed from the event size.
    [3] durations: list of floats
        -- List of avalanche durations.
        -- The duration is the amount of time the velocity is strictly above the threshold.
        -- For example, an avalanche with velocity profile [0,1,2,3,2,1,0] has a duration of 5 timesteps.
    [4] st: list of ints
        -- List of avalanche start indices.
        -- E.g. st[8] is the index on the displacement where the 9th avalanche occurred.
        -- The start index is the first index that the velocity is above the threshold.
    [5] en: list of ints
        -- List of avalanche end indices.
        -- E.g. en[8] will be the index on the displacement where the 9th avalanche ends.
        -- The end index is the first index after the velocity is below the threshold (accounting for Python index counting).
    """
    arr = np.array

    # Alert the user if no data is given
    if displacement is None and velocity is None:
        print("Error! Give at least one of velocity or displacement.")
        return -1

    # Set the threshtype value based on input or alert the user if no valid choice is given
    if threshtype == 'mean':
        threshtype = 1
    elif threshtype == 'median':
        threshtype = 2
    elif threshtype == 'sliding median' or 'sliding_median':
        threshtype = 3
    else:
        print("Error! Parameter threshtype must be one of the following: \n mean \n median \n sliding median")
        return -1

    # Create the time vector if none is given
    if time is None and displacement is None:
        time = np.arange(0, len(velocity))
    if time is None and velocity is None:
        time = np.arange(0, len(displacement))

    # Make window_size the nearest odd number for compatability with sliding_median.
    window_size = window_size + (1 - window_size % 2)

    # Velocity input.
    # st and en indices are the start and end in velocity-land.
    # The time vector is shifted forward by 1 index so the displacement and time are same length.
    is_integrated = 0
    if displacement is None:
        # Displacement start index is index_velocity_begins + 1.
        # Displacement end index is index_velocity_ends + 1.
        displacement = scipy.integrate.cumulative_trapezoid(velocity, time)
        displacement = np.append([displacement[0], displacement[0]], displacement)
        dt = np.median(np.diff(time))
        # Make time and displacement the same length.
        time = np.append(time[0], time + dt)
        is_integrated = 1

    # Displacement input.
    # st and en indices are start and end in displacement-land.
    if velocity is None:
        # Displacement start index is index_velocity_begins + 1.
        # Displacement end index is index_displacement_ends.
        velocity = np.diff(displacement) / np.diff(time)

    # If looking at drops, invert the signal.
    displacement = ((-1) ** drops) * arr(displacement)
    velocity = ((-1) ** drops) * arr(velocity)

    outs = get_slips_core(displacement, velocity, time, threshold, mindrop, is_integrated, threshtype, window_size)

    return outs


def get_slips_core(smoothed, deriv, time, threshhold, mindrop, is_integrated, threshtype, window_size):
    """
    Tyler's get slips, but takes in displacement and velocity.
    Vectorized for speed and various off-by-one errors are fixed.
    Time and smoothed should be length N, deriv should be length N-1.
    """
    # Like standard deviation but for median
    mad = scipy.stats.median_abs_deviation
    sliding_median = scipy.ndimage.median_filter
    arr = np.array
    median = np.median
    mean = np.mean
    ones = np.ones
    zeros = np.zeros
    std = np.std
    diff = np.diff
    append = np.append
    where = np.where

    smoothed = arr(smoothed)
    deriv = arr(deriv)

    # We now take a numeric derivative and get an average
    if threshtype == 1:
        diff_avg = mean(deriv)
        diff_avg = ones(len(deriv)) * diff_avg
    # Do pure median
    # Works better at getting the true size
    elif threshtype == 2:
        diff_avg = median(deriv)
        diff_avg = ones(len(deriv)) * diff_avg
    else:
        diff_avg = sliding_median(deriv, window_size, mode='nearest')
        diff_avg[:window_size // 2] = diff_avg[window_size // 2]
        diff_avg[-window_size // 2:] = diff_avg[-window_size // 2]

    # We now set the minimum slip rate for it to be considered an avalanche
    if threshhold == -1:
        min_diff = zeros(len(deriv))
    else:
        if threshtype == 1:
            min_diff = diff_avg + std(deriv) * threshhold
        else:
            # Works for both cases of sliding median and overall median.
            # New code using median average difference (see https://en.wikipedia.org/wiki/Median_absolute_deviation)
            # Estimates the standard deviation of the noisy part of the data by assuming noise is normally distributed and using relationship between MAD and standard deviation for normal data.
            min_diff = diff_avg + 1.4826 * threshhold * mad(deriv - diff_avg)

    # Now we see if a slip is occurring, i.e. if the derivative is above the minimum value.

    # Shift by one index forward
    slips = append(0, diff(1 * (deriv > min_diff)))
    # Velocity start index (first index above 0)
    velocity_index_begins = where(slips == 1)[0]
    # Velocity end index (last index above 0)
    velocity_index_ends = where(slips == -1)[0]

    # We must consider the case where we start or end on an avalanche.
    # This checks if this is case and if so makes the start or end of the data a start or end of an avalanche.
    if velocity_index_begins.size == 0:
        velocity_index_begins = arr([0])
    if velocity_index_ends.size == 0:
        velocity_index_ends = arr([len(time) - 1])
    if velocity_index_begins[-1] >= velocity_index_ends[-1]:
        velocity_index_ends = append(velocity_index_ends, len(time) - 1)
    if velocity_index_begins[0] >= velocity_index_ends[0]:
        velocity_index_begins = np.insert(velocity_index_begins, 0, 0)

    # Correcting for if displacement is integrated or not.

    # In all cases, the displacement index is one more than velocity index begins.
    displacement_index_begins = velocity_index_begins
    # + is_integrated
    displacement_index_ends = velocity_index_ends

    # Reported start and end index different depending on if displacement or velocity is given.
    index_begins = is_integrated * velocity_index_begins + (1 - is_integrated) * displacement_index_begins
    index_ends = is_integrated * velocity_index_ends + (1 - is_integrated) * displacement_index_ends

    # Now we see if the drops were large enough to be considered an avalanche.
    # Avalanche duration calculated from signal input.
    # First-order approximation assuming the background velocity is constant throughout the avalanche.
    mindrop_correction = diff_avg[index_begins] * (time[index_ends - is_integrated] - time[index_begins]) * int(threshhold != -1)

    index_av_begins = index_begins[mindrop < (smoothed[displacement_index_ends] - smoothed[displacement_index_begins] - mindrop_correction)]
    index_av_ends = index_ends[mindrop < (smoothed[displacement_index_ends] - smoothed[displacement_index_begins] - mindrop_correction)]

    # Finally we use these indices to get the durations and sizes of the events, accounting for diff().
    slip_durations = time[index_av_ends] - time[index_av_begins]
    dt = median(diff(time))

    # Mindrop correction term.
    # Term goes to 0 if avalanche is a single time step, leading to LeBlanc et al 2016 definition of size.
    duration_calculation = time[index_av_ends - is_integrated] - time[index_av_begins]

    # Increment size calculation by 1 if the displacement was integrated to account for cumtrapz()
    is_step = 1 * ((slip_durations <= dt) * is_integrated * (index_av_ends < len(smoothed) - 1))

    # Sizes are more accurately reported by only correcting for background rate, not the rate + threshold
    slip_sizes = smoothed[index_av_ends + is_step] - smoothed[index_av_begins] - diff_avg[index_av_begins] * duration_calculation * int(threshhold != -1)
    time_begins = time[index_av_begins]
    time_ends = time[index_av_ends]
    time2 = 0.5 * (time[0:len(time) - 1] + time[1:len(time)])
    # Sampling time
    tsamp = median(diff(time2))

    velocity = []
    times = []
    for k in range(len(index_av_begins)):
        st = index_av_begins[k]
        en = index_av_ends[k]
        mask = np.arange(st, en)
        if st == en:
            mask = st

        # First-order approximation: assume the shape begins and ends at min_diff halfway between the start index and the preceeding index.
        curv = zeros(en - st + 2)
        curt = zeros(en - st + 2)
        curv[1:-1] = deriv[mask] - min_diff[st]
        curt[1:-1] = time2[mask]
        curt[0] = curt[1] - tsamp / 2
        curt[-1] = curt[-2] + tsamp / 2
        velocity.append(list(curv))
        times.append(list(curt))

    return [list(velocity), list(times), list(slip_sizes), list(slip_durations), list(index_av_begins), list(index_av_ends)]


def get_slips_vel(time, velocity, drops=True, threshold=0, mindrop=0, threshtype='median', window_size=101):
    """
    Identical to get_slips_wrap & get_slips_core in terms of parameters and outputs, but specifically designed for velocity signals & to ensure no negative sizes.
    Use the trapezoidal rule + chopping off all parts of the signal less than the threshold (i.e. velocity < threshold*std(data) or whatever) to get a more accurate view of the size of an event.

    Parameters
    ----------
    time: (List or array-like; REQUIRED)
        Time vector in data units.
        Defaults to an index array, i.e., an array ranging from 0 to N - 1, where N is the length of the input data.
    velocity: (List or array-like; REQUIRED)
        Time series data to be analyzed for avalanches IF data is some quantity where at each time a new value is acquired.
        I.e., number of spins flipped in one timestep of the random-field Ising model (RFIM) or the number of cell failures in one timestep in a slip model.
        An avalanche in this perspective is a "slip RATE", in the parlance of the Dahmen Group.
    drops: (Boolean; OPTIONAL)
        Default value is TRUE.
        Whether to scan the time series for drops in the data.
    threshold: (Float; OPTIONAL)
        Default value is 0.
        Number of standard deviations above the average velocity a fluctuation must be before it is considered an event.
        Recommend 0 for most applications.
        Setting this equal to -1 forces a zero-velocity threshold on velocity curve.
            This is useful for simulations since there's little to no noise to be mistaken for an avalanche.
    mindrop: (Float; OPTIONAL)
        Default value is 0.
        Minimum size required to be counted as an event.
        Recommend 0 when starting analysis, but should be set higher when i.e. data culling to find the true value of tau.
    threshtype: (String; OPTIONAL)
        Default value is 'median'.
        What type of threshold to use. Options:
        'median'
            -- Uses the median velocity instead of the mean velocity to quantify the average velocity.
            -- Works best in signals with many excursions (i.e., many avalanches) and is not sensitive to outliers.
            -- Threshold is calculated using median absolute deviation (MAD) with this option instead of standard deviation because it more accurately describes the dispersion of the noise fluctuations while ignoring the avalanches.
        'mean'
            -- This is the traditional method.
            -- The threshold is compared to the mean velocity, (displacement[end]-displacement[start])/(time[end]-time[start]).
            -- Threshold is calculated using the standard deviation with this setting.
            -- This method has some major issues in non-simulation environments, as the standard deviation is very sensitive to large excursions from the noise floor
        'sliding_median'
            -- Uses a sliding median of width window_length to obtain a sliding median estimate of the background velocity.
            -- Useful when the background rate of the process is not constant over the course of the experiment.
            -- Threshold is calculated using median absolute deviation with this option and follows the change in average velocity while accurately describing the dispersion of the noise.
    window_size: (Int; OPTIONAL)
        Default value is 1.
        The window size, in datapoints, to be used when calculating the sliding median.
        Should be set to be much longer than the length of an avalanche in your data.
        Jordan found setting window_size = 3% the length of the data (in one case) was good, but this value can be anywhere from just a few hundred datapoints (very short avalanches, many datapoints) to up to 10-20% of the total length of the signal (when the data are shorter but contain several very long avalanches).
        This is worth playing with!

    Returns
    -------
    [0] velocity: list of lists
        -- A list of avalanche velocity curves.
        -- E.g. v[8] will be the velocity curve for the 9th avalanche.
        -- Velocity curve will always begin and end with 0s because the velocity had to cross 0 both times in its run.
    [1] times: list of lists
        -- A list of avalanche time curves.
        -- E.g. t[8] will be the corresponding times at which v[8] velocity curve occured.
        -- The time curve will have the 0s in the velocity curve occur at t_start - ts/2 and t_end + ts/2 as a 0-order estimate of when the curve intersected with zero.
    [2] sizes: list of floats
        -- List of avalanche sizes.
        -- The size is defined as the amount the displacement changes while the velocity is strictly above the threshold.
        -- Each size is corrected by the background rate.
        -- That is, (background rate)*(duration) is removed from the event size.
    [3] durations: list of floats
        -- List of avalanche durations.
        -- The duration is the amount of time the velocity is strictly above the threshold.
        -- For example, an avalanche with velocity profile [0,1,2,3,2,1,0] has a duration of 5 timesteps.
    [4] st: list of ints
        -- List of avalanche start indices.
        -- E.g. st[8] is the index on the displacement where the 9th avalanche occurred.
        -- The start index is the first index that the velocity is above the threshold.
    [5] en: list of ints
        -- List of avalanche end indices.
        -- E.g. en[8] will be the index on the displacement where the 9th avalanche ends.
        -- The end index is the first index after the velocity is below the threshold (accounting for Python index counting).
    """
    trapz = scipy.integrate.trapezoid
    # Like standard deviation but for median
    mad = scipy.stats.median_abs_deviation
    sliding_median = scipy.ndimage.median_filter
    ones = np.ones
    append = np.append
    diff = np.diff
    where = np.where
    arr = np.array
    insert = np.insert
    zeros = np.zeros
    median = np.median

    std = lambda x: 1.4826 * mad(x)
    if threshtype == 'median':
        avg = np.median
    if threshtype == 'mean':
        avg = np.mean
        std = np.std
    cutoff_velocity = (avg(velocity) + std(velocity) * threshold * (int(threshold != -1))) * ones(len(velocity))
    if threshtype == 'sliding_median':
        cutoff_velocity = sliding_median(velocity, window_size, mode='nearest')
        cutoff_velocity[:window_size // 2] = cutoff_velocity[window_size // 2]
        cutoff_velocity[-window_size // 2:] = cutoff_velocity[-window_size // 2]
        cutoff_velocity = cutoff_velocity + std(velocity) * threshold * (int * threshold != -1)

    # Treat the velocity by removing the trend such that its centered around zero.
    deriv = velocity - cutoff_velocity
    if drops:
        deriv = -velocity

    # Search for rises in the deriv curve.
    # Set all parts of the curve with velocity less than zero to be equal to zero.
    deriv[deriv < 0] = 0
    # Get the slips
    slips = append(0, diff(1 * (deriv > 0)))
    # Velocity start index (first index above 0)
    index_begins = where(slips == 1)[0]
    # Velocity end index (last index above 0)
    index_ends = where(slips == -1)[0]

    if index_begins.size == 0:
        index_begins = arr([0])
    if index_ends.size == 0:
        index_ends = arr([len(time) - 1])
    if index_begins[-1] >= index_ends[-1]:
        index_ends = append(index_ends, len(time) - 1)
    if index_begins[0] >= index_ends[0]:
        index_begins = insert(index_begins, 0, 0)

    # Get the possible sizes
    possible_sizes = zeros(len(index_begins))
    possible_durations = zeros(len(index_begins))
    for i in range(len(index_begins)):
        st = index_begins[i]
        en = index_ends[i]
        trapz_st = max([st - 1, 0])
        trapz_en = min(en + 1, len(deriv))
        possible_sizes[i] = trapz(deriv[trapz_st:trapz_en], time[trapz_st:trapz_en])
        possible_durations[i] = time[en] - time[st]

    idxs = where(possible_sizes > mindrop)[0]
    sizes = possible_sizes[idxs]
    durations = possible_durations[idxs]
    index_av_begins = index_begins[idxs]
    index_av_ends = index_ends[idxs]

    time2 = 0.5 * (time[0:len(time) - 1] + time[1:len(time)])
    # Sampling time
    tsamp = median(diff(time2))
    time2 = append(time2, time2[-1] + tsamp)
    velocity = []
    times = []
    for k in range(len(index_av_begins)):
        st = index_av_begins[k]
        en = index_av_ends[k]
        mask = np.arange(st, en)
        if st == en:
            mask = st

        # First-order approximation: assume the shape begins and ends at min_diff halfway between the start index and the preceeding index.
        curv = zeros(en - st + 2)
        curt = zeros(en - st + 2)
        curv[1:-1] = deriv[mask]
        curt[1:-1] = time2[mask]
        curt[0] = curt[1] - tsamp / 2
        curt[-1] = curt[-2] + tsamp / 2
        velocity.append(list(curv))
        times.append(list(curt))

    return [list(velocity), list(times), list(sizes), list(durations), list(index_av_begins), list(index_av_ends)]
