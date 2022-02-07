import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as arima
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import functions as func
import os


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
    return current_diff

def nth_seasonal_difference(data, n, period):
    '''
    ### Finds nth seasonal difference of seasonal/periodic data
    data: list or 1-D array, y-data
    n: int, number of times to difference data
    period: int, period of seasonal data (in number of data points)
    # Note: to find nth time derivative, simply divide by (dt)^n
    '''
    current_diff = pd.DataFrame(data)
    for i in range(n):
        current_diff = current_diff - current_diff.shift(period)
    return current_diff

def ARIMA(data, order=(1,2,1)):
    '''
    ### fits ARIMA model to data
    data: list or 1-D array, y-data
    order: vector of form (p,d,q), ARIMA parameters:
        p: int, lag order
        d: int, degree of differencing
        q: int, size of moving average window
    '''
    model=arima(data, order=order)
    return model.fit()

def SARIMAX(data, order, seasonal_order):
    '''
    ### fits Seasonal ARIMA model to data
    data: list or 1-D array, y-data
    order: vector of form (p,d,q), ARIMA parameters:
        p: int, lag order
        d: int, degree of differencing
        q: int, size of moving average window
    seasonal_order: vector of form (P,D,Q,period)
        P: int, seasonal lag order
        D: int, seasonal degree of differencing
        Q: int, size of moving average window
        period: int, data period as number of points
    '''
    model=sm.tsa.statespace.SARIMAX(data, order=order, seasonal_order=seasonal_order)
    # model=arima(data, order=order, seasonal_order=seasonal_order)
    return model.fit()

def arima_predictor(model, start, end, dynamic=True):
    nan_array = [np.NaN] * (start - 1)
    predicted = model.predict(start=start, end=end, dynamic=dynamic)
    predicted = np.concatenate((nan_array, predicted))
    return predicted

def pathmaker(filename):
    file_name = filename
    fullfilename = os.path.join(os.path.dirname(__file__), file_name)
    return(fullfilename)

'''
temperatures = pd.read_csv(pathmaker("Testing/daily-min-temperatures.csv"), header=0)
model=sm.tsa.statespace.SARIMAX(temperatures["Temp"],order=(1, 1, 1),seasonal_order=(1,1,1,365))
results=model.fit()

predicted=results.predict(start=3000,end=3600,dynamic=True)
'''