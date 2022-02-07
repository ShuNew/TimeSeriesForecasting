import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import scipy.signal
from scipy.optimize import curve_fit 
import statsmodels.api as sm

# Averages & interpolation
#**************************** These are not necessary, use the pandas-ta package instead **********************************#
def SMA(data, window):
    '''
    ### Simple Moving Average
    data: list or 1-D array, y-data (floats)
    window: int, number of points to average over
    '''
    np.array(data)
    m_avg = []
    for n in range(len(data)):
        if n < window:
            m_avg.append(None)
        else:
            avg = np.sum(data[(n-window):n]) / window
            m_avg.append(avg)
    return m_avg

# Other
def func_dev_upper(data, window, function_data, scaling_factor=1):
    '''
    ### Pseudo Upper Bollinger Band
    # Uses rolling_RMSD function
    data: list or 1-D array, y-data (floats)
    window: int, number of points to average over
    function_data: 1-D array or list, function data to plot (floats)
    scaling_factor: float, scaling factor for band (default=1)
    '''
    function_data = np.array(function_data)
    moving_dev = np.array(rolling_RMSD(data, window, function_data))
    diff = abs(len(moving_dev) - len(function_data))

    if len(function_data) < len(moving_dev):
        moving_dev = moving_dev[diff:]
    elif len(function_data) > len(moving_dev):
        function_data = function_data[diff:]

    FDU = function_data + scaling_factor * moving_dev
    if diff > 0:
        none_list = [np.NaN] * diff
        FDU = np.concatenate((none_list, FDU))

    return FDU

def func_dev_lower(data, window, function_data, scaling_factor=1):
    '''
    ### Pseudo Lower Bollinger Band
    # Uses rolling_RMSD function
    data: list or 1-D array, y-data (floats)
    window: int, number of points to average over
    function_data: 1-D array or list, function data to plot (floats)
    scaling_factor: float, scaling factor for band (default=1)
    '''
    function_data = np.array(function_data)
    moving_dev = np.array(rolling_RMSD(data, window, function_data))
    diff = abs(len(moving_dev) - len(function_data))

    if len(function_data) < len(moving_dev):
        moving_dev = moving_dev[diff:]
    elif len(function_data) > len(moving_dev):
        function_data = function_data[diff:]

    FDL = function_data - scaling_factor * moving_dev
    if diff > 0:
        none_list = [np.NaN] * diff
        FDL = np.concatenate((none_list, FDL))
        
    return FDL
    
### non-labelled functions
# Curve Fits
def best_fit_linear(data, time, get_params=False):
    '''
    ### Linear Trend Line
    data: list or 1-D array, y-data (floats)
    time: list or 1-D array, time data (datetime)
    get_params: bool, if True then get best fit parameters, if False then get best fit line and label (default: False)
    '''
    def f(x, m, b):
        return m * x + b
    time = mdates.date2num(time)
    popt, pcov = curve_fit(f, time, data)
    m, b = popt[0], popt[1]
    lin_trend = m * time + b

    if get_params == True:
        return popt, pcov
    else:
        return lin_trend

def stdev_fit_linear(data, time, window, get_params=False):
    '''
    ### Linear Trend Line for rolling stdev
    data: list or 1-D array, y-data (floats)
    time: list or 1-D array, time data (datetime)
    get_params: bool, if True then get best fit parameters, if False then get best fit line and label (default: False)
    '''
    rolling_stdev = pd.Series(data).rolling(window).std()
    return best_fit_linear(rolling_stdev, time, get_params)

# Periodic Functions
def period_DFT(data, time_btwn_pts=1, sigma_threshold=3.0):
    '''
    ### Returns approximate period of seasonal/periodic data using Discrete Fourier Transform
    # Note: requires regularly spaced data points
    data: list or 1-D array, y-data
    time_btwn_pts: float, number of seconds between consecutive data points (must be constant)
    sigma_threshold: float, threshold for peak height in number of standard deviations (default = 3)
    '''
    sf = 1/time_btwn_pts
    freq, power = scipy.signal.periodogram(data, sf)

    threshold = sigma_threshold * np.std(power)
    peaks = scipy.signal.find_peaks(power, height=threshold)[0]

    period = []
    peak_heights = np.zeros(len(power))
    for peak in peaks:
        period.append(float(1/freq[peak]))
        peak_heights[peak] = power[peak]

    avg_peak_strength_ratio = np.mean(peak_heights)/np.mean(power - peak_heights)

    return period, avg_peak_strength_ratio

def period_autocorr(data, time_btwn_pts=1, sigma_threshold=3.0):
    '''
    ### Returns approximate period of seasonal/periodic data using Discrete Fourier Transform of Autocorrelation plot
    # Note: requires regularly spaced data points
    data: list or 1-D array, y-data
    time_btwn_pts: float, number of seconds between consecutive data points (must be constant)
    sigma_threshold: float, threshold for peak height in number of standard deviations (default = 3)
    '''
    autocorr_list = sm.tsa.acf(data, nlags=len(data), fft=False)
    return period_DFT(autocorr_list, time_btwn_pts, sigma_threshold)

# Net Profit
def profit(buy, sell, fee, fee_type_abs=True):
    '''
    ### Calculate net profit for a given trading strategy
    buy: list or 1-D array, list of buy prices (floats)
    sell: list or 1-D array, list of sell prices (floats)
    fee: float or np.array of format[buy_fee_1, ..., buy_fee_n, sell_fee_1, ..., sell_fee_m], transaction fee, 
            if fee_type_abs==True then absolute amount, if fee_type_abs==False then fraction of buy/sell price
    fee_type_abs: bool, sets whether fee is absolute or fraction of buy/sell amount (default=True)
    '''
    if fee_type_abs==True:
        profitability = np.sum(sell) - np.sum(buy) - (len(buy) + len(sell)) * fee
    else:
        profitability = np.sum(sell) - np.sum(buy) - (len(buy) + len(sell)) * (fee * np.concatenate((buy, sell)))
    return profitability

# Smoothing and Filters
def savgol(data, window=11, max_poly_deg=5, rmsd_threshold=0.02, threshold_absolute=False):
    '''
    ### Apply a Savitzky-Golay filter to data
    # Instead of specifying order of polynomial, specify max RMSD within window
    data: list or 1-D array, y-data (floats)
    window: odd int >=3, length of filter window
    max_poly_deg: int, max size of polynomial to fit
    rmsd_threshold: float, maximum RMSD from filtered data, absolute value or relative to data size (default=0.02)
    threshold_absolute: bool, sets whether rmsd_threshold is absolute or relative (default=False)
    # Note: if max_poly_deg is too large the function might break (recommend <= 15)
    # Personal Note: This is generally better than MA because it does not have a time lag
    '''
    data = np.array(data)

    poly_deg = min(window-1, max_poly_deg)
    def filtered_data(poly_deg):
        return scipy.signal.savgol_filter(data, window, poly_deg)
    
    def filtered_rmsd(poly_deg):
        return rolling_RMSD(data, window, filtered_data(poly_deg))
    
    if threshold_absolute==True:
        while (max(filtered_rmsd(poly_deg)) <= rmsd_threshold) and (poly_deg >= 1):
            poly_deg -=1
    elif threshold_absolute==False:
        while (max(filtered_rmsd(poly_deg)/data[:-window]) <= rmsd_threshold) and (poly_deg >= 1):
            poly_deg -=1

    if poly_deg == window-1:
        return filtered_data(poly_deg)
    else:
        return filtered_data(poly_deg+1)

# Loss Function
def loss_abs_error(data, function_data):
    '''
    ### Absolute error from data
    data: list or 1-D array, y-data (floats)
    function_data: 1-D array or list, function data to plot (floats)
    # Note: data and function_data must be of same size
    '''
    data = np.array(data)
    function_data = np.array(function_data)
    return np.abs(data - function_data)

def loss_abs_savgol(data, function_data, window, max_poly_deg=5, rmsd_threshold=0.02, threshold_absolute=False):
    '''
    ### Savitzky-Golay filter applied to absolute error function
    data: list or 1-D array, y-data (floats)
    function_data: 1-D array or list, function data to plot (floats)
    window: odd int >=3, length of filter window
    # Note: data and function_data must be of same size
    '''
    error = loss_abs_error(data, function_data)
    smooth_error = savgol(error, window, max_poly_deg, rmsd_threshold, threshold_absolute)
    return smooth_error

# Datetime
def time_to_points(time_list, start_time, dt):
    pass

def points_to_time(point_list, start_time, dt):
    pass

# Other
def rolling_RMSD(data, window, function_data):
    '''
    ### Rolling Root-mean-square deviation of data from function
    # Possible use case as a loss function or 'pseudo-Bollinger band' for an arbitrary function
    data: list or 1-D array, y-data (floats)
    window: int, number of points to average over
    function_data: 1-D array or list, function data to plot (floats)
    *** Must have len(data) >= len(function_data) ***
    '''
    diff = len(data) - len(function_data)
    data = data[diff:]

    rmsd_list = []
    for n in range(len(data)-window):
        moving_sum = np.sum((function_data[n:n+window] - data[n:n+window])**2)
        RMSD = np.sqrt(moving_sum/window)
        rmsd_list.append(RMSD)
    
    if diff > 0:
        none_list = [None] * diff
        rmsd_list.insert(0, none_list)
    return np.array(rmsd_list)

def lin_trend_remover(data, time, window):
    '''
    ### Removes linear trends, both additive and multiplicative
    data: list or 1-D array, y-data (floats)
    time: list or 1-D array, time data (datetime)
    window: int, number of points to take stdev over
    # Subtracts linear best fit, divides linear best fit of rolling stdev
    '''
    data = data - best_fit_linear(data, time)
    data = data / stdev_fit_linear(data, time, window)
    return data

def stationarity_test(data, split_number=5, absolute=True):
    '''
    ### Determine stationarity of data
    data: list or 1-D array, y-data (floats)
    split_number: int, number of arrays to split data into (default=5)
    absolute: bool, determines if mean and variance stdev values are absolute or relative to the mean
    '''
    data = np.array(data)

    if len(data) % split_number == 0:
        split_data = np.split(data, split_number)
    else:
        split_index = int(np.floor(len(data)/split_number) + (len(data)%split_number))
        split_data_1, split_data_2 = np.split(data, [split_index])
        split_data_2 = np.split(split_data_2, split_number - 1)
        for n in range(split_number-1):
            split_data_2[n] = np.ndarray.tolist(split_data_2[n])
        split_data_2.insert(0, split_data_1)
        split_data = split_data_2

    mean_list = []
    var_list = []

    for index in range(split_number):
        mean_list.append(np.mean(split_data[index]))
        var_list.append(np.var(split_data[index]))

    autocorr_list = sm.tsa.acf(data, nlags=len(data), fft=False)
    pos_autocorr = 0
    neg_autocorr = 0
    for i in autocorr_list:
        if i > 0:
            pos_autocorr += 1
        if i < 0:
            neg_autocorr += 1
    
    mean_std = np.std(mean_list)
    var_std = np.std(var_list)
    autocorr_diff = abs(pos_autocorr - neg_autocorr)

    if absolute==True:
        return mean_std, var_std, autocorr_diff
    else:
        stdev = np.std(data)
        return mean_std/stdev, var_std/stdev**2, autocorr_diff/len(autocorr_list)
