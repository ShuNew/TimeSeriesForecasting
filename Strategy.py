import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import datetime
import scipy.signal
import re
import os

### Filepath Maker
def pathmaker(filename):
    fullfilename = os.path.join(os.path.dirname(__file__), filename)
    return(fullfilename)

def pathmaker_absolute(filepath, filename):
    filepath = re.sub(r'\\', r'\/', filepath)
    fullfilename = os.path.join(filepath, filename)
    return(fullfilename)

### CSV File
def create_csv(filepath, name):
    filepath = re.sub(r'\\', r'\/', filepath)
    fullfilename = os.path.join(filepath, name + ".csv")
    BuySellData = pd.DataFrame(columns=['Timestamp', 'Buy_Prices', 'Sell_Prices', 'num_owned', 'current_profit'])
    BuySellData.loc[len(BuySellData)] = ([np.NaN, np.NaN, np.NaN, int(0), 0.0])
    BuySellData.to_csv(fullfilename, index=False)
    return

### Buy and Sell
def Buy(csv_file, price, datetime, fee=0, fee_absolute=True, api=None, act=False):

    if act==True:
        # Buy API Here
        pass
    
    BuySellData = pd.DataFrame(pd.read_csv(csv_file, header=0))
    BuySellData.loc[len(BuySellData)] = ([datetime, price, np.NaN, (BuySellData.loc[len(BuySellData)-1,'num_owned'] + 1), np.NaN])

    num_owned = BuySellData.loc[len(BuySellData)-1, 'num_owned']
    Buy_Prices = np.array(BuySellData['Buy_Prices'][~np.isnan(BuySellData['Buy_Prices'])])
    Sell_Prices = np.array(BuySellData['Sell_Prices'][~np.isnan(BuySellData['Sell_Prices'])])
    if fee_absolute==False:
        BuySellData.loc[len(BuySellData)-1,'current_profit'] = (np.sum(Sell_Prices) + num_owned*price) - (np.sum(Buy_Prices) + np.sum(Buy_Prices*fee))
    else:
        BuySellData.loc[len(BuySellData)-1,'current_profit'] = (np.sum(Sell_Prices) + num_owned*price) - (np.sum(Buy_Prices) + len(Buy_Prices)*fee)

    BuySellData.to_csv(csv_file, index=False)
    return

def Sell(csv_file, price, datetime, fee=0, fee_absolute=True, api=None, act=False):

    if act==True:
        # Sell API Here
        pass

    BuySellData = pd.DataFrame(pd.read_csv(csv_file, header=0))
    BuySellData.loc[len(BuySellData)] = ([datetime, np.NaN, price, (BuySellData.loc[len(BuySellData)-1,'num_owned'] - 1), np.NaN])
    
    num_owned = BuySellData.loc[len(BuySellData)-1, 'num_owned']
    Buy_Prices = np.array(BuySellData['Buy_Prices'][~np.isnan(BuySellData['Buy_Prices'])])
    Sell_Prices = np.array(BuySellData['Sell_Prices'][~np.isnan(BuySellData['Sell_Prices'])])
    if fee_absolute==False:
        BuySellData.loc[len(BuySellData)-1,'current_profit'] = (np.sum(Sell_Prices) + num_owned*price) - (np.sum(Buy_Prices) + np.sum(Buy_Prices*fee))
    else:
        BuySellData.loc[len(BuySellData)-1,'current_profit'] = (np.sum(Sell_Prices) + num_owned*price) - (np.sum(Buy_Prices) + len(Buy_Prices)*fee)

    BuySellData.to_csv(csv_file, index=False)
    return

### Sample Strategies
def SMA_strategy(data, csv_file, budget=1000000.0, fee=0, api=None, window=10, max_loss=1, threshold=0, fee_absolute=True, max_loss_absolute=False, threshold_absolute=True, act=False):
    '''
    ### Buy if SMA crosses (data + threshold), Sell if MA crosses (data + threshold)
    data: 2-D array, [timestamps, y-data] [datetime, floats]
    csv_file: filepath (str), filepath (including file name) for Buy/Sell data
    budget: float, total budget in given currency
    fee: float, fee per transaction in given currency
    window: odd int >=3, length of filter window
    max_loss: None or float, maximum permitted loss before selling (make sure to put a decimal point to force number to be a float)
    max_loss_absolute: bool, sets whether max_loss is absolute or relative to buy price (default=True)
    threshold_absolute: bool, sets whether threshold is absolute or relative to current price (default=True)
    act: bool, sets whether to act on a Buy/Sell signal or just add to a list
    '''
    Time = data[0]
    Price = data[1]

    BuySellData = pd.DataFrame(pd.read_csv(csv_file, header=0))
    num_owned = BuySellData.loc[len(BuySellData)-1, 'num_owned']
    Buy_Prices = np.array(BuySellData['Buy_Prices'][~np.isnan(BuySellData['Buy_Prices'])])
    Sell_Prices = np.array(BuySellData['Sell_Prices'][~np.isnan(BuySellData['Sell_Prices'])])
    last_buy = Buy_Prices[-1]

    if threshold_absolute==False:
        threshold = threshold * Price[-1]
    if max_loss_absolute==False:
        max_loss = max_loss * last_buy
    if fee_absolute==False:
        total_spent = np.sum(Buy_Prices) + np.sum(Buy_Prices*fee) - np.sum(Sell_Prices)
    else:
        total_spent = np.sum(Buy_Prices) + len(Buy_Prices)*fee - np.sum(Sell_Prices)
    budget_remaining = budget - total_spent

    ### Strategy
    SMA = ta.sma(Price, length=window)

    check = 0 # Check to prevent multiple buys without crossing over threshold
    
    if (check==1) and (SMA[-1] < Price[-1]):
        check = 0
    if (check==-1) and (SMA[-1] > Price[-1]):
        check = 0

    if ((Price[-1] + fee) <= budget_remaining) and (check!=1) and (SMA[-1] > (Price[-1]+threshold)):
        Buy(csv_file, Price[-1], Time[-1], fee, fee_absolute, api, act)

    if num_owned > 0:
        if Price[-1] <= max_loss:
            Sell(csv_file, Price[-1], Time[-1], fee, fee_absolute, api, act)
        if (check!=-1) and (SMA[-1] < (Price[-1]-threshold)):
            Sell(csv_file, Price[-1], Time[-1], fee, fee_absolute, api, act)
    
    return

### My Strategies
def my_strategy(*args):
    pass
