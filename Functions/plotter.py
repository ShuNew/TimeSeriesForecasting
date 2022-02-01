import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Datetime format
dt_format = '%Y-%m-%d %H:%M:%S'

# List of 99 random colors
np.random.seed(99)
color_list = []
for color in range(99):
    RGB = (np.random.random(), np.random.random(), np.random.random())
    color_list.append(RGB)

def lineplotter(PriceData, columns, time_column="timestamp", title="Price Plot", split=False, start="all", end="all", functions=None):
    '''
    ### PriceData: pandas DataFrame, at least one column must be time

    columns: list of strings, names of columns to plot
    time_column: string, name of time column
    title: string, title of plot
    split: bool, if True: show functions on separate subplots, if False: show functions on same plot (default: False)
    start: datetime object, start date/time of plot (default: show all)
    start: datetime object, end date/time of plot (default: show all)
    (optional) functions: 1-D array or list, function data to plot

    ### Additional Notes: each function in "functions" should be a tuple of form (data, label)
    '''
    time = PriceData[time_column]
    myFmt = mdates.DateFormatter(dt_format)

    n = 0 # axis/color index

    if split==False:
        fig, ax = plt.subplots()

        for col in columns:
            ax.plot(time, PriceData[str(col)], label=str(col), color=color_list[n])
            n += 1
        
        if functions != None:
            for function in functions:
                func_data, label = function
                ax.plot(time, func_data, label=str(label), color=color_list[n])
                n += 1
        ax.legend()
    
    if split==True:
        col_rows = len(columns)
        func_rows = len(functions)
        fig, ax = plt.subplots(nrows=col_rows+func_rows, sharex=True)

        for col in columns:
            ax[n].plot(time, PriceData[str(col)], label=str(col), color=color_list[n])
            ax[n].legend()
            n += 1

        if functions:
            for function in functions:
                func_data, label = function
                ax[n].plot(time, func_data, label=str(label), color=color_list[n])
                ax[n].legend()
                n += 1
        
        ax[0].xaxis.set_major_formatter(myFmt)

    fig.autofmt_xdate()

    fig.suptitle(title)
    plt.xlabel("Time Stamp (YYYY-MM-DD hh:mm:ss)")
    plt.ylabel("Price ($)")

    if not isinstance(start, str):
        plt.xlim(left=start)
    if not isinstance(end, str):
        plt.xlim(right=end)

    plt.show()

def barplotter_time(PriceData, name, column):
    pass

def barplotter_frequency(PriceData, name, column):
    pass
