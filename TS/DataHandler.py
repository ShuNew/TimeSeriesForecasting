# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:31:50 2020

@author: Somnath
"""

# Libraries
import pandas as pd
import numpy as np

import Configuration as config

# Names used in the code 
close_col = "close"
high_col = "high"
low_col = "low"
open_col = "open"
volume_col = "volume"
no_col = "No"
date_col = "DataDate"
timestamp_col = "timestamp"
range_col = "Range"
typicalPrice_col = "TypicalPrice"

def GetFileName(FolderName, symbol, exchange,  interval):
    filename = FolderName + symbol + "_" + exchange
    if(len(interval)>0):
       filename = filename + "_" + interval
    return  filename  + ".csv"

def GetIntradayFileName(symbol, exchange,  interval):
    return GetFileName(config.IntraDayFolder, symbol, exchange, interval)

def GetEndofdayFileName(symbol, exchange):
    return GetFileName(config.DatbaseFolder, symbol, exchange)

def GetPrice(Ticker, Exchange):
    filename = GetEndofdayFileName(Ticker, Exchange)
    data = pd.read_csv(filename, parse_dates=True)
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col)
    return data

def GetClosePrice(Ticker, Exchange):
    data = GetPrice(Ticker, Exchange)
    return data[[close_col]]

def GetAggregatedPrice(Ticker, Exchange, Frequency):
    data = GetPrice(Ticker, Exchange)
    
    if(Frequency=='W'):
        data_agg = data.resample('W-FRI')
    if(Frequency=='M'):
        # data_agg = data.resample('MS', loffset=pd.Timedelta(days=data.index[0].day) - 1)
        data_agg = data.resample('M')
    
    data_agg_close = data_agg[[close_col]].last()
    data_agg_close = data_agg_close.rename(columns={close_col: close_col + "_" + Frequency})

    data_agg_high = data_agg[[high_col]].max()
    data_agg_high = data_agg_high.rename(columns={high_col: high_col + "_" + Frequency})

    data_agg_low = data_agg[[low_col]].min()
    data_agg_low = data_agg_low.rename(columns={low_col: low_col + "_" + Frequency})

    data_agg_open = data_agg[[open_col]].min()
    data_agg_open = data_agg_open.rename(columns={open_col: open_col + "_" + Frequency})

    data_agg_volume = data_agg[[volume_col]].sum()
    data_agg_volume = data_agg_volume.rename(columns={volume_col: volume_col + "_" + Frequency})

    return data_agg_close.join(data_agg_high, how='left').join(data_agg_low, how='left').join(data_agg_open, how='left').join(data_agg_volume, how='left')

def GetWeeklyPrice(Ticker, Exchange):
    return GetAggregatedPrice(Ticker, Exchange, 'W')

def GetMonthlyPrice(Ticker, Exchange):
    return GetAggregatedPrice(Ticker, Exchange, 'M')

def AttachData(df1,df2,fillUp):
    combinedata =  df1.join(df2, how='left')
    if(fillUp):
        combinedata = combinedata.interpolate(method='linear', limit_direction='forward', axis=0)
    return combinedata

def downLoadIntradayPrice(symbol, exchange,  interval):
    # Plumbing Parameters
    function = "TIME_SERIES_INTRADAY"
    apikey = "O4TCJD77T51J5IA3"
    url = 'https://www.alphavantage.co/query?function='+str(function)+'&symbol='+str(symbol)+'&interval='+str(interval)+'&apikey='+str(apikey)+'&datatype=csv'

    #Downlad
    fullfilename = GetIntradayFileName(symbol, exchange,  interval)
    data = pd.read_csv(url)
    data.to_csv(fullfilename, index=False)

def GetIntradayPriceData(symbol, exchange,  interval):
    fullfilename = GetIntradayFileName(symbol, exchange, interval)
    data = pd.read_csv(fullfilename, parse_dates=True)
    
    #Repare Date
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.sort_values(by=timestamp_col)
    #data = data.reset_index(drop=True)
    data.insert(loc=0, column=no_col, value=np.arange(len(data)))
    return(data)

def GetTS(PriceDataFrame, priceColumn):
    return PriceDataFrame[priceColumn].squeeze()

def AddRange(data):
    data[range_col] = data[high_col] - data[low_col]
    return data

def AddTypicalPrice(data):
    data[typicalPrice_col] = (data[high_col] + data[low_col] + data[close_col])/3
    return data    








