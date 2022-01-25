# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 23:32:10 2022

@author: Somnath
"""
import pandas as pd
import numpy as np
import Configuration as config
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Use this to toggle the plot window mode
# %matplotlib auto
# %matplotlib inline

close_col = "close"
high_col = "high"
low_col = "low"
no_col = "No"
timestamp_col = "timestamp"
MACD_col = "MACD"
MACD_MA_col = "MACD_MA"
timestamp_col = "timestamp"

def GetFileName(symbol, exchange,  interval):
    return config.IntraDayFolder + symbol + "_" + exchange + "_" + interval + ".csv"

def downLoadData(symbol, exchange,  interval):
    # Plumbing Parameters
    function = "TIME_SERIES_INTRADAY"
    apikey = "O4TCJD77T51J5IA3"
    url = 'https://www.alphavantage.co/query?function='+str(function)+'&symbol='+str(symbol)+'&interval='+str(interval)+'&apikey='+str(apikey)+'&datatype=csv'

    #Downlad
    fullfilename = GetFileName(symbol, exchange,  interval)
    data = pd.read_csv(url)
    data.to_csv(fullfilename, index=False)
    
def LoadData(symbol, exchange,  interval):
    fullfilename = GetFileName(symbol, exchange,  interval)
    data = pd.read_csv(fullfilename, parse_dates=True)
    
    #Repare Date
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data = data.sort_values(by=timestamp_col)
    #data = data.reset_index(drop=True)
    data.insert(loc=0, column=no_col, value=np.arange(len(data)))
    return(data)

def GetPriceTS(PriceDataFrame, priceColumn):
    return PriceDataFrame[priceColumn].squeeze()

def SetFormat(axisObject):
    axisObject.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    axisObject.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    axisObject.legend()
    
Ticker = "IBM"
Exchange = "NYSE"
Frequency = "5min"

# downLoadData(Ticker, Exchange , Frequency)
PriceData = LoadData(Ticker, Exchange, Frequency )
print(PriceData)

ClosePriceData = GetPriceTS(PriceData, close_col )
fig = plt.figure()
plt.plot(PriceData[no_col],PriceData[close_col])
SetFormat(plt)
plt.show()

ClosePriceData.describe()

MA_period = 10
Growth_period = 10

SMA_col = "SMA_"+ str(MA_period)
EMA_col = "EMA_"+ str(MA_period)
WMA_col = "WMA_"+ str(MA_period)
MLR_col = "MLR_"+ str(MA_period)


# Simple Moving Average
ClosePriceMA = ClosePriceData.rolling(window=MA_period).mean()
ClosePriceMA = ClosePriceMA.rename(SMA_col)

# Exponential Moving Average
ClosePriceEMA = ClosePriceData.ewm(span=MA_period, adjust=False).mean()
ClosePriceEMA = ClosePriceEMA.rename(EMA_col)

# MACD
MACD1_period = 12
MACD2_period = 26
MACD_MA_period = 9

MACD1_col = "EMA_"+ str(MACD1_period)
MACD2_col = "EMA_"+ str(MACD2_period)

MACD1EMA = ClosePriceData.ewm(span=MACD1_period, adjust=False).mean()
MACD2EMA = ClosePriceData.ewm(span=MACD2_period, adjust=False).mean()

#Combine all Data
CombineAnalysisData = PriceData
CombineAnalysisData["Range"] = CombineAnalysisData[high_col] - CombineAnalysisData[low_col]
CombineAnalysisData["TypicalPrice"] = (CombineAnalysisData[high_col] + CombineAnalysisData[low_col] + CombineAnalysisData[close_col])/3
CombineAnalysisData["PreviousClose"] =  CombineAnalysisData[close_col].shift(1)
CombineAnalysisData["MovingGrowth_"+str(Growth_period)] = CombineAnalysisData[close_col] / ClosePriceData.shift(Growth_period-1)-1
CombineAnalysisData[SMA_col] = ClosePriceMA
CombineAnalysisData[EMA_col] = ClosePriceEMA
CombineAnalysisData[MACD1_col] = MACD1EMA
CombineAnalysisData[MACD2_col] = MACD2EMA
CombineAnalysisData[MACD_col] = CombineAnalysisData[MACD1_col] - CombineAnalysisData[MACD2_col]
CombineAnalysisData[MACD_MA_col] = CombineAnalysisData[MACD_col].rolling(window=MACD_MA_period).mean()

# WMA
weights = np.arange(1, MA_period+1, 1, dtype=int)
sum_weights = np.sum(weights)

CombineAnalysisData[WMA_col] = (CombineAnalysisData[close_col]
    .rolling(window=MA_period)
    .apply(lambda x: np.sum(weights*x) / sum_weights, raw=False)
)

# Moving Linear Regression
CombineAnalysisData[MLR_col] = 3* CombineAnalysisData[WMA_col] - 2* CombineAnalysisData[SMA_col]

print(CombineAnalysisData)

# Combine all MA
fig = plt.figure()

x_ax = PriceData["No"]

plt.plot(x_ax,PriceData[close_col], label = close_col )
plt.plot(x_ax,PriceData[SMA_col], label = SMA_col )
plt.plot(x_ax,PriceData[EMA_col], label = EMA_col )
plt.plot(x_ax,PriceData[WMA_col], label = WMA_col )
plt.plot(x_ax,PriceData[MLR_col], label = MLR_col )

# Linear Regression
sns.regplot(x=no_col, y=close_col, data=CombineAnalysisData, scatter=False)

SetFormat(plt)
plt.show()

x=CombineAnalysisData[no_col]

fig, axs = plt.subplots(3)
fig.suptitle('Price : ' + Ticker + "-" + Exchange + " (" + Frequency + ")")

axs[0].plot(x,CombineAnalysisData[close_col],label = close_col)
axs[0].plot(x,CombineAnalysisData[SMA_col],label = SMA_col)
axs[0].plot(x,CombineAnalysisData[EMA_col],label = EMA_col)
axs[0].plot(x,CombineAnalysisData[WMA_col],label = WMA_col)
axs[0].plot(x,CombineAnalysisData[MLR_col],label = MLR_col)

# Linear Regression
sns.regplot(x=no_col, y=close_col, data=CombineAnalysisData, ax = axs[0], scatter=False)

SetFormat(axs[0])

# Growth
axs[1].plot(CombineAnalysisData["MovingGrowth_"+str(Growth_period)],label="Growth_"+str(Growth_period))
axs[1].axhline(y=0.0, color='r', linestyle='-')
SetFormat(axs[1])

# MACD
axs[2].plot(x,CombineAnalysisData[MACD_col],label = MACD_col)
axs[2].plot(x,CombineAnalysisData[MACD_MA_col],label = MACD_MA_col)
axs[2].axhline(y=0.0, color='r', linestyle='-')
SetFormat(axs[2])
plt.show()

# Decomposition
str_multiplicative = "multiplicative"
str_additive = "additive"

decomposeresult = seasonal_decompose(ClosePriceData, model=str_additive,period=MA_period)
print(decomposeresult.trend)
print(decomposeresult.seasonal)
print(decomposeresult.resid)
print(decomposeresult.observed)
decomposeresult.plot()
plt.show()