import numpy as np
import matplotlib.dates as mdates
import scipy.signal
from scipy.optimize import curve_fit 
import statsmodels.api as sm

### labelled functions
# Averages & interpolation
#**************************** These are not necessary, use the pandas-ta package instead **********************************#
def SMA(data, length):
    '''
    ### Simple Moving Average (label = "SMA")
    data: list or 1-D array, y-data (floats)
    length: int, number of points to average over
    '''
    label = "SMA"
    np.array(data)
    m_avg = []
    for n in range(len(data)):
        if n < length:
            m_avg.append(None)
        else:
            avg = np.sum(data[(n-length):n]) / length
            m_avg.append(avg)
    return m_avg, label

# Curve Fits
def best_fit_linear(data, time, get_params=False):
    '''
    ### Linear Trend Line (label = "Linear Trend Line")
    data: list or 1-D array, y-data (floats)
    time: list or 1-D array, time data (datetime)
    get_params: bool, if True then get best fit parameters, if False then get best fit line and label (default: False)
    '''
    label = "Linear Trend Line"
    def f(x, m, b):
        return m * x + b
    time = mdates.date2num(time)
    popt, pcov = curve_fit(f, time, data)
    m, b = popt[0], popt[1]
    lin_trend = m * time + b

    if get_params == True:
        return popt, pcov
    else:
        return lin_trend, label

# Other
def func_dev_upper(data, length, function_data, scaling_factor=1):
    '''
    ### Pseudo Upper Bollinger Band (label = "Upper Band")
    # Uses rolling_RMSD function
    data: list or 1-D array, y-data (floats)
    length: int, number of points to average over
    function_data: 1-D array or list, function data to plot (floats)
    scaling_factor: float, scaling factor for band (default=1)
    '''
    label = "Upper Band"
    function_data = np.array(function_data)
    moving_dev = np.array(rolling_RMSD(data, length, function_data))
    diff = abs(len(moving_dev) - len(function_data))

    if len(function_data) < len(moving_dev):
        moving_dev = moving_dev[diff:]
    elif len(function_data) > len(moving_dev):
        function_data = function_data[diff:]

    FDU = function_data + scaling_factor * moving_dev
    if diff > 0:
        none_list = [np.NaN] * diff
        FDU = np.concatenate((none_list, FDU))

    return FDU, label

def func_dev_lower(data, length, function_data, scaling_factor=1):
    '''
    ### Pseudo Lower Bollinger Band (label = "Lower Band")
    # Uses rolling_RMSD function
    data: list or 1-D array, y-data (floats)
    length: int, number of points to average over
    function_data: 1-D array or list, function data to plot (floats)
    scaling_factor: float, scaling factor for band (default=1)
    '''
    label = "Lower Band"
    function_data = np.array(function_data)
    moving_dev = np.array(rolling_RMSD(data, length, function_data))
    diff = abs(len(moving_dev) - len(function_data))

    if len(function_data) < len(moving_dev):
        moving_dev = moving_dev[diff:]
    elif len(function_data) > len(moving_dev):
        function_data = function_data[diff:]

    FDL = function_data - scaling_factor * moving_dev
    if diff > 0:
        none_list = [np.NaN] * diff
        FDL = np.concatenate((none_list, FDL))
        
    return FDL, label 
    
### non-labelled functions
# Periodic Functions
def nl_apx_period(data, time_btwn_pts, sigma_threshold=3.0):
    '''
    ### Returns approximate period of seasonal/periodic data
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
    for peak in peaks:
        period.append(float(1/freq[peak]))

    return period

def nl_periodicity(data, seconds_btwn_pts):
    pass

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
def savgol(data, window_length=11, rmsd_threshold=0.02, threshold_absolute=False):
    '''
    ### Apply a Savitzky-Golay filter to data
    # Instead of specifying order of polynomial, specify max RMSD within window
    data: list or 1-D array, y-data (floats)
    window_length: odd bool >=3, length of filter window
    rmsd_threshold: float, maximum RMSD from filtered data, absolute value or relative to data size (default=0.02)
    threshold_absolute: bool, sets whether rmsd_threshold is absolute or relative (default=False)
    # Note: if window_length is too large the function might break (recommend <= 15% of data length)
    '''
    data = np.array(data)

    poly_deg = window_length-1
    def filtered_data(poly_deg):
        return scipy.signal.savgol_filter(data, window_length, poly_deg)
    
    def filtered_rmsd(poly_deg):
        return rolling_RMSD(data, window_length, filtered_data(poly_deg))
    
    if threshold_absolute==True:
        while (max(filtered_rmsd(poly_deg)) <= rmsd_threshold) and (poly_deg >= 1):
            poly_deg -=1
    elif threshold_absolute==False:
        while (max(filtered_rmsd(poly_deg)/data[:-window_length]) <= rmsd_threshold) and (poly_deg >= 1):
            poly_deg -=1

    if poly_deg == window_length-1:
        return filtered_data(poly_deg)
    else:
        return filtered_data(poly_deg+1)

# Other
def rolling_RMSD(data, length, function_data):
    '''
    ### Rolling Root-mean-square deviation of data from function
    # Possible use case as a loss function or 'pseudo-Bollinger band' for an arbitrary function
    data: list or 1-D array, y-data (floats)
    length: int, number of points to average over
    function_data: 1-D array or list, function data to plot (floats)
    *** Must have len(data) >= len(function_data) ***
    '''
    diff = len(data) - len(function_data)
    data = data[diff:]

    rmsd_list = []
    for n in range(len(data)-length):
        moving_sum = np.sum((function_data[n:n+length] - data[n:n+length])**2)
        RMSD = np.sqrt(moving_sum/length)
        rmsd_list.append(RMSD)
    
    if diff > 0:
        none_list = [None] * diff
        rmsd_list.insert(0, none_list)
    return np.array(rmsd_list)

def stationarity_test(data, split_number=5, lag=1, absolute=True):
    '''
    ### Determine stationarity of data
    data: list or 1-D array, y-data (floats)
    split_number: int, number of arrays to split data into (default=5)
    lag: int, lag for autocorrelation calculation (default=1)
    absolute: bool, determines if mean and variance stdev values are absolute or relative to the mean
    '''
    data = np.array(data)

    if len(data) % split_number == 0:
        split_data = np.split(data, split_number)
    else:
        split_index = np.floor(len(data)/split_number) + (len(data)%split_number)
        split_data_1, split_data_2 = np.split(data, [split_index])
        split_data_2 = np.split(split_data_2, split_number - 1)
        split_data = np.concatenate((split_data_1, split_data_2))

    mean_list = []
    var_list = []
    autocorr_list = []

    for index in range(np.shape(split_data)[0]):
        mean_list.append(np.mean(split_data[index]))
        var_list.append(np.var(split_data[index]))
        autocorr_list.append(sm.tsa.acf(split_data[index], nlags=lag, fft=False)[lag])

    mean_std = np.std(mean_list)
    var_std = np.std(var_list)
    autocorr_std = np.std(autocorr_list)

    if absolute==True:
        return mean_std, var_std, autocorr_std
    else:
        mean = np.mean(mean_list)
        return mean_std/mean, var_std/mean, autocorr_std
