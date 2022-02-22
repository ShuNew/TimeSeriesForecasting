import numpy as np
import pandas as pd
import talib as ta
import matplotlib.dates as mdates
import scipy.signal
from scipy.optimize import curve_fit 
import statsmodels.api as sm
from numba import jit, njit
from numba import float64
from numba import int64
from itertools import (takewhile,repeat)

# CSV Handling
# Credit: Michael Bacon on Stack Overflow, https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
def CSVlinecount(file):
    f = open(file, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )

# NaN and Array Handling
def nan_remover(data):
    '''
    ### Removes non-numeric values from list or array of numbers
    data: 1-D list or array, data to remove non-numeric values from
    # Outputs: (nan_index_list, NaNfree_list)
    # Note: make sure to use nan_replacer() to replace NaN values when finished
    '''
    data = pd.DataFrame(data)
    NaNified_list = np.array(data.apply(pd.to_numeric, errors='coerce').iloc[:,0])
    try:
        nan_index_list = np.where(np.isnan(NaNified_list)==True)[0]
    except:
        nan_index_list = []
    if len(nan_index_list) >= 1:
        NaNfree_list = np.delete(NaNified_list, nan_index_list)
    else:
        NaNfree_list = NaNified_list
    return NaNfree_list, nan_index_list

def nan_replacer(data, nan_index_list):
    '''
    ### Adds back NaN values to list or array of numbers
    data: 1-D list or array, data to add NaN values to
    '''
    if len(nan_index_list) >= 1:
        nan_arr = [np.NaN] * len(nan_index_list)
        data = np.insert(data, nan_index_list, nan_arr)
    return data

# Credit: chrisaycock on Stack Overflow; https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def arr_shift(arr, num, fill_value=np.NaN):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

# Math
def nth_difference(data, n):
    '''
    ### Finds nth difference of data
    data: list or 1-D array, y-data
    n: int, number of times to difference data
    # Note: to find nth time derivative, simply divide by (dt)^n
    '''
    current_diff = np.array(data)
    for i in range(n):
        current_diff = np.diff(current_diff)

    nan_num = len(data) - len(current_diff)
    nan_arr = [np.NaN] * nan_num

    current_diff = np.concatenate([nan_arr, current_diff])

    return current_diff

def round_nth_decimal(data, n=0, roundtype="down"):
    '''
    ### Round number or array to nth decimal
    n: int, number of decimal points to round to
    roundtype: str, "nearest" or "up" or "down"
    '''
    if hasattr(data, '__len__') and (not isinstance(data, str)):
        data = np.array(data)
    if roundtype=="nearest":
        data = np.round(data, n)
    else:
        data = data * 10**n
        if roundtype=="up":
            data = np.ceil(data)
        elif roundtype=="down":
            data = np.floor(data)
        data = data * 10**(-n)
    return data

def round_to_increment(data, increment=1, roundtype="down"):
    '''
    ### Round number or array to nearest increment
    n: int, number of decimal points to round to
    roundtype: str, "nearest" or "up" or "down"
    '''
    if hasattr(data, '__len__') and (not isinstance(data, str)):
        data = np.array(data)
    
    data = data/increment
    if roundtype=="nearest":
        data = np.round(data, 0)
    if roundtype=="up":
        data = np.ceil(data)
    elif roundtype=="down":
        data = np.floor(data)
    data = data*increment

    return data

# Averages & interpolation
#***These are not necessary, use the pandas-ta or talib packages instead ****#
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

# Lag-corrected smoothing
# These either do not exist or are slow on pandas-ta

# Credit: Alexander McFarlane on Stack Overflow, https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
@jit((float64[:], int64), nopython=True, nogil=True)
def fast_EWMA(arr_in, window):
    """Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    window : int64
        The decay window, or 'span'

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(span=10, adjust=True).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 10), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma

def ZLEMA(data, window=1):
    '''
    ### "Zero Lag" Exponential Moving Average
    data: list or 1-D array, y-data (floats)
    window: int, number of points to average over
    weight, float between 0 and 1, weighting for EMA, if left blank it will default to 2/(window + 1)
    # See https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
    '''
    data = np.array(data)
    # data, nan_index = nan_remover(data)

    lag = int(np.floor((window-1)/2))
    EMA_data = 2*data - arr_shift(data, lag)
    # ZLEMA_data = fast_EWMA(EMA_data, window)
    ZLEMA_data = ta.EMA(EMA_data, timeperiod=window)
    # ZLEMA_data = nan_replacer(EMA_data, nan_index)
    return ZLEMA_data



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
    max_poly_deg: int, max degree of polynomial to fit
    rmsd_threshold: float, maximum RMSD from filtered data, absolute value or relative to data size (default=0.02)
    threshold_absolute: bool, sets whether rmsd_threshold is absolute or relative (default=False)
    # Note: if max_poly_deg is too large the function might break (recommend <= 15)
    # Personal Note: This is generally better than MA because it does not have a time lag
    '''
    data, nan_index_list = nan_remover(data)
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
        return nan_replacer(filtered_data(poly_deg), nan_index_list)
    else:
        return nan_replacer(filtered_data(poly_deg+1), nan_index_list)

def savgol_gen(data, window=11, poly_deg=5):
    '''
    ### Apply a Savitzky-Golay filter to data
    # This function only deals with NaN values, no optimization of poly_deg
    data: list or 1-D array, y-data (floats)
    window: odd int >=3, length of filter window
    poly_deg: int, degree of polynomial to fit
    # Note: if poly_deg is too large the function might break (recommend <= 15)
    # Personal Note: This is generally better than MA because it does not have a time lag
    '''
    data, nan_index_list = nan_remover(data)
    data = np.array(data)
    return nan_replacer(scipy.signal.savgol_filter(data, window, poly_deg), nan_index_list)

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
@njit
def RMSD_loop(data, window, function_data, rmsd_list):
    for n in range(len(data)-window):
        N = n+window
        moving_sum = np.sum((function_data[n:N] - data[n:N])**2)
        RMSD = np.sqrt(moving_sum/window)
        rmsd_list[N] = RMSD
    return rmsd_list

def rolling_RMSD(data, window, function_data):
    '''
    ### Rolling Root-mean-square deviation of data from function
    # Possible use case as a loss function or 'pseudo-Bollinger band' for an arbitrary function
    data: list or 1-D array, y-data (floats)
    window: int, number of points to average over
    function_data: 1-D array or list, function data to plot (floats)
    *** Must have len(data) >= len(function_data) ***
    '''
    rmsd_list = np.empty_like(data)
    rmsd_list[:] = np.NaN
    
    rmsd_list = RMSD_loop(data, window, function_data, rmsd_list)
    return np.array(rmsd_list)

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

    split_data = np.array_split(data, split_number)

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

# Modified versions of pandas-ta functions

# Modified from pandas-ta.jma to use numpy arrays (and work with numba)
@njit
def jma_loop(m, close, volty, v_sum, sum_length, length1, pow1, bet, beta, pr, kv, det0, det1, ma2, jma, ma1, uBand, lBand):
    for i in range(1, m):
        price = close[i]

        # Price volatility
        del1 = price - uBand
        del2 = price - lBand
        volty[i] = max(abs(del1),abs(del2)) if abs(del1)!=abs(del2) else 0

        # Relative price volatility factor
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[max(i - sum_length, 0)]) / sum_length
        avg_volty = np.mean(v_sum[max(i - 65, 0):i + 1])
        d_volty = 0 if avg_volty ==0 else volty[i] / avg_volty
        r_volty = max(1.0, min(np.power(length1, 1 / pow1), d_volty))

        # Jurik volatility bands
        pow2 = np.power(r_volty, pow1)
        kv = np.power(bet, np.sqrt(pow2))
        uBand = price if (del1 > 0) else price - (kv * del1)
        lBand = price if (del2 < 0) else price - (kv * del2)

        # Jurik Dynamic Factor
        power = np.power(r_volty, pow1)
        alpha = np.power(beta, power)

        # 1st stage - prelimimary smoothing by adaptive EMA
        ma1 = ((1 - alpha) * price) + (alpha * ma1)

        # 2nd stage - one more prelimimary smoothing by Kalman filter
        det0 = ((price - ma1) * (1 - beta)) + (beta * det0)
        ma2 = ma1 + pr * det0

        # 3rd stage - final smoothing by unique Jurik adaptive filter
        det1 = ((ma2 - jma[i - 1]) * (1 - alpha) * (1 - alpha)) + (alpha * alpha * det1)
        jma[i] = jma[i-1] + det1
    return jma

def jma(close, length=None, phase=0.0, offset=None, **kwargs):
    """Indicator: Jurik Moving Average (JMA)"""
    # Validate Arguments
    _length = int(length) if length and length > 0 else 7
    close = close if len(close) > _length else None
    try:
        offset = int(offset)
    except:
        offset = 0
    if close is None: return

    # Define base variables
    jma = np.zeros_like(close)
    volty = np.zeros_like(close)
    v_sum = np.zeros_like(close)

    kv = det0 = det1 = ma2 = 0.0
    jma[0] = ma1 = uBand = lBand = close[0]

    # Static variables
    sum_length = 10
    length = 0.5 * (_length - 1)
    pr = 0.5 if phase < -100 else 2.5 if phase > 100 else 1.5 + phase * 0.01
    length1 = max((np.log(np.sqrt(length)) / np.log(2.0)) + 2.0, 0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * np.sqrt(length)
    bet = length2 / (length2 + 1)
    beta = 0.45 * (_length - 1) / (0.45 * (_length - 1) + 2.0)

    m = close.shape[0]
    jma = jma_loop(m, close, volty, v_sum, sum_length, length1, pow1, bet, beta, pr, kv, det0, det1, ma2, jma, ma1, uBand, lBand)

    # Remove initial lookback data and convert to pandas frame
    jma[0:_length - 1] = np.NaN

    # Offset
    if offset != 0:
        jma = arr_shift(jma, offset)

    # # Handle fills
    # if "fillna" in kwargs:
    #     jma.fillna(kwargs["fillna"], inplace=True)
    # if "fill_method" in kwargs:
    #     jma.fillna(method=kwargs["fill_method"], inplace=True)

    # # Name & Category
    # jma.name = f"JMA_{_length}_{phase}"
    # jma.category = "overlap"

    return np.array(jma)


jma.__doc__ = \
"""Jurik Moving Average Average (JMA)

Mark Jurik's Moving Average (JMA) attempts to eliminate noise to see the "true"
underlying activity. It has extremely low lag, is very smooth and is responsive
to market gaps.

Sources:
    https://c.mql5.com/forextsd/forum/164/jurik_1.pdf
    https://www.prorealcode.com/prorealtime-indicators/jurik-volatility-bands/

Calculation:
    Default Inputs:
        length=7, phase=0

Args:
    close (pd.Series): Series of 'close's
    length (int): Period of calculation. Default: 7
    phase (float): How heavy/light the average is [-100, 100]. Default: 0
    offset (int): How many lengths to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""