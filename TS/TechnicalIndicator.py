# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 00:47:09 2020

@author: Somnath
"""
import numpy as np

# Preious Value
def PreviousValue(data, period, ValueColumn, PreviousColumn):
    data[PreviousColumn] =  data[ValueColumn].shift(period)
    return data

# Rolling Growth
def RollingGrowth(data, period, ValueColumn, GrowthColumn):
    data[GrowthColumn] =  data[ValueColumn] / data[ValueColumn].shift(period-1) - 1
    return data

# Rolling Sum
def RollingSum(data, period, ValueColumn, RSColumn):
    data[RSColumn] =  data[ValueColumn].rolling(period, center=False).sum()
    return data

# Simple Moving Average
def SMA(data, period, ValueColumn, MAColumn):
    data[MAColumn] =  data[ValueColumn].rolling(period, center=False).mean()
    return data

# Weighted Moving Average
def WMA(data, period, ValueColumn, MAColumn):
    weights = np.arange(1, period+1, 1, dtype=int)
    sum_weights = np.sum(weights)
    
    data[MAColumn] = (data[ValueColumn]
        .rolling(window=period)
        .apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
    )

    return data

# Exponential Moving Average
def EMA(data, period, ValueColumn, MAColumn):
    data[MAColumn] =  data[ValueColumn].ewm(span=period, adjust=False).mean()
    return data

# Double Exponential Moving Average
def DEMA(data, period, ValueColumn, MAColumn):
    strPeriod = str(period)
    EMA_col = "EMA_"+ strPeriod
    EMA_EMA_col = "EMA_EMA_" + strPeriod
    
    data = EMA(data, period, ValueColumn, EMA_col)
    data = EMA(data, period, EMA_col, EMA_EMA_col)
    
    data[MAColumn] = 2*data[EMA_col]- data[EMA_EMA_col]
    return data

# Triple Exponential Moving Average
def TEMA(data, period, ValueColumn, MAColumn):
    strPeriod = str(period)
    EMA_col = "EMA_" + strPeriod
    EMA_EMA_col = "EMA_EMA_" + strPeriod
    EMA_EMA_EMA_col = "EMA_EMA_EMA_" + strPeriod
    
    data = EMA(data, period, ValueColumn, EMA_col)
    data = EMA(data, period, EMA_col, EMA_EMA_col)
    data = EMA(data, period, EMA_EMA_col, EMA_EMA_EMA_col)
    
    data[MAColumn] = 3*(data[EMA_col]- data[EMA_EMA_col]) + data[EMA_EMA_EMA_col]
    return data

# MACD
def MACD(data, MACD1_period, MACD2_period, ValueColumn, MACDColumn,MACD_MA_period):
    MACD1_col = "EMA_"+ str(MACD1_period)
    MACD2_col = "EMA_"+ str(MACD2_period)
    data[MACD1_col] = data[ValueColumn].ewm(span=MACD1_period, adjust=False).mean()
    data[MACD2_col] = data[ValueColumn].ewm(span=MACD2_period, adjust=False).mean()
    data[MACDColumn] = data[MACD1_col] - data[MACD2_col]
    MACD_MA_col = MACDColumn+"_MA"
    data[MACD_MA_col] = data[MACDColumn].rolling(window=MACD_MA_period).mean()

    return data

# Moving Standard Deviation
def MSD(data, period, ValueColumn, SDColumn):
    data[SDColumn] =  data[ValueColumn].rolling(period, center=False).std()
    return data

# Bollinger Bands
def BB(data, period, ValueColumn, BBColumn):
    data = SMA(data,period,ValueColumn,BBColumn)
    SDColumn = "SD_"+ str(period)
    data = MSD(data,period,ValueColumn,SDColumn)
    
    data[BBColumn + "_DN"] =  data[BBColumn] - data[SDColumn]
    data[BBColumn + "_UP"] =  data[BBColumn] + data[SDColumn]
    
    return data

# Moving Linear Regression Slope
def MLRS(data, period, ValueColumn, MLRSColumn):
    SMAColumn = "SMA_"+ str(period)
    data = SMA(data,period,ValueColumn, SMAColumn)
    WMAColumn = "WMA_"+ str(period)
    data = WMA(data,period,ValueColumn, WMAColumn)
    
    data[MLRSColumn] = 6*(data[WMAColumn] - data[SMAColumn])/(period-1)

    return data

# Moving Linear Regression
def MLR(data, period, ValueColumn, MLRColumn):
    SMAColumn = "SMA_"+ str(period)
    data = SMA(data,period,ValueColumn, SMAColumn)
    WMAColumn = "WMA_"+ str(period)
    data = WMA(data,period,ValueColumn, WMAColumn)
    
    data[MLRColumn] = 3* data[WMAColumn] - 2* data[SMAColumn]

    return data






    
