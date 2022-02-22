from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

# Datetime format
dt_format = '%Y-%m-%d %H:%M:%S'

# List of 99 random colors
np.random.seed(99)
color_list = []
for color in range(99):
    RGB = (np.random.random(), np.random.random(), np.random.random())
    color_list.append(RGB)

def lineplotter(PriceData, columns=[], time_column="timestamp", title="Price Plot", split=False, start="all", end="all", functions=None):
    '''
    ### PriceData: pandas DataFrame, at least one column must be time

    columns: list of strings, names of columns to plot (NOT including time_column)
    time_column: string, name of time column
    title: string, title of plot
    split: bool, if True: show functions on separate subplots, if False: show functions on same plot (default: False)
    start: datetime object, start date/time of plot (default: show all)
    start: datetime object, end date/time of plot (default: show all)
    (optional) functions: 2-tuple (1D list, str), function data and labels to plot 
    # Additional Notes: each function in "functions" should be a tuple of form (data, label)
    '''
    PriceData = pd.DataFrame(PriceData)

    columns.insert(0, time_column)

    df = PriceData[columns]
    if functions != None:
        for function in functions:
            func_data, label = function
            df[label] = func_data

    colors = color_list[:len(df.columns)]
    
    if split==True:
        subplot_param = True
    else:
        subplot_param = False
    
    if any(char.isdigit() for char in start):
        starttime = pd.to_datetime(start)
    else:
       starttime = pd.to_datetime(min(df[time_column]))

    if any(char.isdigit() for char in end):
        endtime = pd.to_datetime(end)
    else:
        endtime = pd.to_datetime(max(df[time_column]))

    df[time_column] = pd.to_datetime(df[time_column])
    df_time = df.set_index(time_column)

    start_index = df.loc[df.index.unique()[df_time.index.unique().get_loc(starttime, method='nearest')]].name
    end_index = df.loc[df.index.unique()[df_time.index.unique().get_loc(endtime, method='nearest')]].name

    if(start_index > end_index):
        start_copy = np.copy(start_index)
        start_index = end_index
        end_index = start_copy

    plot_df = df.iloc[start_index:end_index+1]

    ax = plot_df.plot(title=title, subplots=subplot_param, x=time_column, y=df.columns.tolist()[1:], color=colors)
        
    # ax.set_xticklabels([pandas_datetime.strftime(dt_format) for pandas_datetime in plot_df[time_column]])
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time Stamp")
    plt.ylabel("Price ($)")

    plt.show()

def barplotter_time(PriceData, name, column):
    pass

def barplotter_frequency(PriceData, name, column):
    pass

# df = pd.DataFrame()
# date_df = pd.DataFrame({
#              'year': [2022]*50,
#              'month': [1]*50,
#              'day': [1]*50,
#              'hour': [0]*50,
#              'minute': [0]*50,
#              'second': np.arange(0, 50, 1)
#             })
# date_df.loc[len(date_df)] = [2022,1,1,0,0,49]

# df['time'] = pd.to_datetime(date_df)
# df['b'] = np.random.rand(51)

# print(df)

# lineplotter(df, ['b'], 'time')