# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:14:22 2020

@author: Somnath
"""
import DataHandler as DB
import PlotHandler as plth
import PriceIndicator as PInd
import TechnicalIndicator as TI
import TSModel as TSM


import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def TestingOne() :
    # Check Data
    data = DB.GetPrice("IBM", "NYS")
    print(data.dtypes)
    print(data.head(10))
    
    close = DB.GetClosePrice("IBM", "NYS")
    print(close.head())
    
    plth.TimeSeriesQuickPlot(data,["Close Price_D","High Price_D"])
    plth.TimeSeriesQuickPlot(data,["Close Price_D","High Price_D"],True)
    
    # Show Different Zoom 
    
    # (Last 1 and Half Years)
    plth.TimeSeriesQuickPlot(data,["Close Price_D","High Price_D"],False, ShowFrom = '2019-01-01')
    
    # (2020)
    plth.TimeSeriesQuickPlot(data,["Close Price_D","High Price_D"],False, ShowFrom = '2020')
    plth.TimeSeriesQuickPlot(data,["Close Price_D","High Price_D"],True, ShowFrom = '2020')
    
    # (2020-Jun)
    plth.TimeSeriesQuickPlot(data,["Close Price_D","High Price_D"],False, ShowFrom = '2020-06-01',ShowTill = '2020-06-30')
    plth.TimeSeriesQuickPlot(data,["Close Price_D","High Price_D"],True, ShowFrom = '2020-06-01',ShowTill = '2020-06-30')
    
    # Aggregated Data
    
    # Weekly Data
    weekly_data =  DB.GetWeeklyPrice("IBM", "NYS")
    
    # Monthly Data
    monthy_data =  DB.GetMonthlyPrice("IBM", "NYS")
    
    # combinedata =  data.join(data_weekly_mean, how='left')
    # combinedata = combinedata.interpolate(method='linear', limit_direction='forward', axis=0)
    combinedata = DB.AttachData(data, weekly_data, True);
    
    plth.TimeSeriesQuickPlot(combinedata,["Close Price_D","Close Price_W"],False, ShowFrom = '2020-06-01', ShowTill = '2020-06-30')
    plth.TimeSeriesQuickPlot(combinedata,["Close Price_D","Close Price_W"],False, ShowFrom = '2020')
    
    
    plth.TimeSeriesQuickPlot(monthy_data,["Close Price_M","High Price_M"])
    
    RangeData = PInd.GetPriceRange(data)
    MovementData = PInd.GetPriceMovementPerRecord(RangeData)
    MidPrice = PInd.GetMidPrice(data)
    
    MAData = TI.SMA(data, 7, "Close Price_D", "Close Price_7_MA")
    plth.TimeSeriesQuickPlot(MAData,["Close Price_D","Close Price_7_MA"],False, ShowFrom = '2020')
    
def TestingTwo() :
    Ticker = "IBM"
    Exchange = "NYSE"
    Frequency = "5min"
    
    # downLoadData(Ticker, Exchange , Frequency)
    PriceData = DB.GetIntradayPriceData(Ticker, Exchange, Frequency )
    #print(PriceData)
    
    CombineAnalysisData = PriceData
    CombineAnalysisData = DB.AddRange(CombineAnalysisData)
    CombineAnalysisData = DB.AddTypicalPrice(CombineAnalysisData)
    
    MA_period = 10
    Growth_period = 10
    
    SMA_col = "SMA_"+str(MA_period)
    WMA_col = "WMA_"+str(MA_period)
    MLR_col = "MLR_"+str(MA_period)
    
    EMA_col = "EMA_"+str(MA_period)
    DEMA_col = "DEMA_"+str(MA_period)
    TEMA_col = "TEMA_"+str(MA_period)
    BB_col = "BB_"+str(MA_period)
    
    Growth_col = "Growth_"+str(MA_period)
    MLRS_col = "MLRS_"+str(MA_period)
    
    MACD_col = "MACD_"+str(MA_period)
    MACD_MA_col = MACD_col +"_MA"
    
    CombineAnalysisData = TI.PreviousValue(CombineAnalysisData, MA_period, DB.close_col, "Previous_"+str(MA_period))
    CombineAnalysisData = TI.RollingGrowth(CombineAnalysisData, Growth_period, DB.close_col, Growth_col)
    CombineAnalysisData = TI.SMA(CombineAnalysisData, MA_period, DB.close_col, SMA_col)
    CombineAnalysisData = TI.WMA(CombineAnalysisData, MA_period, DB.close_col, WMA_col)
    CombineAnalysisData = TI.EMA(CombineAnalysisData, MA_period, DB.close_col, EMA_col)
    CombineAnalysisData = TI.DEMA(CombineAnalysisData, MA_period, DB.close_col, DEMA_col)
    CombineAnalysisData = TI.TEMA(CombineAnalysisData, MA_period, DB.close_col, TEMA_col)
    CombineAnalysisData = TI.MLR(CombineAnalysisData, MA_period, DB.close_col, MLR_col)
    CombineAnalysisData = TI.MLRS(CombineAnalysisData, MA_period, DB.close_col, MLRS_col)
    CombineAnalysisData = TI.MACD(CombineAnalysisData, 12,26, DB.close_col, MACD_col,9)
    
    CombineAnalysisData = TI.BB(CombineAnalysisData, MA_period, DB.close_col, "BB_"+str(MA_period))
    
    #print(CombineAnalysisData)
    
    ChartCaption = Ticker + "-" + Exchange + " (" + Frequency + ")"
    index_col = DB.no_col

    Axs = plth.ChartArea()
    Axs.Frames.append(plth.Frame(None, DB.close_col,[DB.close_col,SMA_col,EMA_col,DEMA_col,TEMA_col, WMA_col, MLR_col]))
    plth.PlotTimeSeries(CombineAnalysisData, index_col, ChartCaption +"- [Moving Averages]", Axs)
    
    Axs = plth.ChartArea()
    Axs.Frames.append(plth.Frame(None, "",[DB.close_col, BB_col, BB_col+"_UP", BB_col+"_DN"]))
    plth.PlotTimeSeries(CombineAnalysisData, index_col, ChartCaption +"- [Bolinger Bands]", Axs)

    Axs = plth.ChartArea()
    Axs.Frames.append(plth.Frame(None, "",[DB.close_col]))
    Axs.Frames.append(plth.Frame(0.0, "",[Growth_col]))
    Axs.Frames.append(plth.Frame(0.0, "",[MLRS_col]))
    plth.PlotTimeSeries(CombineAnalysisData, index_col, ChartCaption +"- [MACD]", Axs)


    Axs = plth.ChartArea()
    Axs.Frames.append(plth.Frame(None, "",[DB.close_col]))
    Axs.Frames.append(plth.Frame(0.0, "",[MACD_col, MACD_MA_col]))
    plth.PlotTimeSeries(CombineAnalysisData, index_col, ChartCaption +"- [MACD]", Axs)


    decomposeresult = TSM.decompose(CombineAnalysisData[DB.close_col], TSM.str_additive, MA_period, True)

    print(decomposeresult.trend)
    print(decomposeresult.seasonal)
    print(decomposeresult.resid)
    print(decomposeresult.observed)
    
TestingTwo()

