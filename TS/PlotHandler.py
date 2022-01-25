# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:24:52 2020

@author: Somnath
"""

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Plotting Functions
def TimeSeriesQuickPlot(data, TimeSerieses, SeparatePanel=False, ShowFrom='', ShowTill=''):
    
    TimeSeriesData = data[TimeSerieses]
    
    if(ShowFrom == ''):
        ShowFrom = '1900-01-01'
    if(ShowTill == ''):
        ShowTill = '2100-01-01'
    
    TimeSeriesData = TimeSeriesData.loc[ShowFrom:ShowTill]

    axes = TimeSeriesData.plot(subplots=SeparatePanel)
    recordNo = len(TimeSeriesData.index)
    
    if(SeparatePanel):
        for ax in axes:
            SetFormat(ax,recordNo)
    else:
       SetFormat(axes,recordNo)

    plt.show()

def SetFormat(axisObject, record):
    if(record<60):  
        axisObject.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));
    axisObject.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    axisObject.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

def SetGrid_Legend(axisObject):
    axisObject.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    # Customize the minor grid
    axisObject.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    axisObject.legend()

class Frame:
    def __init__(self, hline, regr_col, Axes):
        self.hline = hline
        self.regr_col = regr_col
        self.Axes = Axes
    
class ChartArea:
    def __init__(self):
        self.Frames = []
    
def drawing(data,tsColumnsInAxes, i, x, drawobject):
    for ts in tsColumnsInAxes:
        drawobject.plot(x,data[ts],label = ts)
        SetGrid_Legend(drawobject)    

def PlotTimeSeries(data, index_col, ChartCaption, ChartFrames):
    x=data[index_col]

    axisnos = len(ChartFrames.Frames)

    if(axisnos==1):
        fig = plt.figure()
        frms = ChartFrames.Frames[0]
        drawing(data, frms.Axes, 0, x, plt)
        regr_col = frms.regr_col
        
        if(len(regr_col)>0):
            sns.regplot(x=x, y=regr_col, data=data, scatter=False)
        hline = frms.hline
        if(hline!=None):
            plt.axhline(y=hline, color='r', linestyle='-')
    else:
        fig, axs = plt.subplots(axisnos)
        for i in range(axisnos):
            frms = ChartFrames.Frames[i]
            ax =  axs[i]
            drawing(data, frms.Axes, i, x, ax)
            regr_col = frms.regr_col
            
            if(len(regr_col)>0):
                sns.regplot(x=x, y=regr_col, data=data, ax=ax, scatter=False)

            hline = frms.hline
            
            if(hline!=None):
                axs[i].axhline(y=hline, color='r', linestyle='-')

    fig.suptitle(ChartCaption)
    
    plt.show()