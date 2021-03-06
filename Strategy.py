import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import datetime
import scipy.signal
import re
import os
import itertools
from collections import deque

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
    fullfilename = os.path.join(filepath, name)
    BuySellData = pd.DataFrame(columns=['Timestamp', 'Buy_Prices', 'Sell_Prices', 'Buy_amount', 'Sell_amount', 'fee', 'cumulative_spent', 'cumulative_sold', 'num_owned', 'current_profit'])
    BuySellData.loc[len(BuySellData)] = ([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, 0.0, 0.0, 0.0, 0.0])
    BuySellData.to_csv(fullfilename, index=False)
    return

### Buy and Sell
def Buy(csv_file, price, datetime, amount=1, fee=0, fee_absolute=True, api=None, act=False):

    if act==True:
        # Buy API Here
        # api parameter should be of form (api_name, security, other_args)
        pass
    
    BuySellData = pd.DataFrame(pd.read_csv(csv_file, header=0))
    BuySellData.loc[len(BuySellData)] = ([datetime, price, np.NaN, amount, np.NaN, fee,\
                                        np.NaN, (BuySellData.loc[len(BuySellData)-1,'cumulative_sold']), (BuySellData.loc[len(BuySellData)-1,'num_owned'] + amount), np.NaN])

    if fee_absolute==False:
        BuySellData.loc[len(BuySellData)-1,'cumulative_spent'] = BuySellData.loc[len(BuySellData)-2,'cumulative_spent'] + amount*price*(1+fee)
    else:
        BuySellData.loc[len(BuySellData)-1,'cumulative_spent'] = BuySellData.loc[len(BuySellData)-2,'cumulative_spent'] + amount*price + fee

    BuySellData.loc[len(BuySellData)-1,'current_profit'] = BuySellData.loc[len(BuySellData)-1,'cumulative_sold'] - BuySellData.loc[len(BuySellData)-1,'cumulative_spent'] + price * BuySellData.loc[len(BuySellData)-1,'num_owned']

    BuySellData.to_csv(csv_file, index=False)
    return

def Sell(csv_file, price, datetime, amount=1, fee=0, fee_absolute=True, api=None, act=False):

    if act==True:
        # Sell API Here
        # api parameter should be of form (api_name, security, other_args)
        pass

    BuySellData = pd.DataFrame(pd.read_csv(csv_file, header=0))
    BuySellData.loc[len(BuySellData)] = ([datetime, np.NaN, price, np.NaN, amount, fee,\
                                        np.NaN, np.NaN, (BuySellData.loc[len(BuySellData)-1,'num_owned'] - amount), np.NaN])

    if fee_absolute==False:
        BuySellData.loc[len(BuySellData)-1,'cumulative_spent'] = BuySellData.loc[len(BuySellData)-2,'cumulative_spent'] + amount*price*fee
        BuySellData.loc[len(BuySellData)-1,'cumulative_sold'] = BuySellData.loc[len(BuySellData)-2,'cumulative_sold'] + amount*price*(1+fee)
    else:
        BuySellData.loc[len(BuySellData)-1,'cumulative_spent'] = BuySellData.loc[len(BuySellData)-2,'cumulative_spent'] + fee
        BuySellData.loc[len(BuySellData)-1,'cumulative_sold'] = BuySellData.loc[len(BuySellData)-2,'cumulative_sold'] + amount*price + fee

    BuySellData.loc[len(BuySellData)-1,'current_profit'] = BuySellData.loc[len(BuySellData)-1,'cumulative_sold'] - BuySellData.loc[len(BuySellData)-1,'cumulative_spent'] + price * BuySellData.loc[len(BuySellData)-1,'num_owned']

    BuySellData.to_csv(csv_file, index=False)
    return

### Sample Strategies
def SMA_strategy(data, csv_file, budget=1000000.0, dynamic_budget_alloc=1, buyamount=1, sellamount="all", buyfee=0, sellfee=0, api=None, window=10, max_loss=1, threshold=0, fee_absolute=True, max_loss_absolute=False, threshold_absolute=True, act=False):
    '''
    ### Buy if SMA crosses (data + threshold), Sell if MA crosses (data + threshold)
    data: 2-D array, [timestamps, y-data] [datetime, floats]
    csv_file: filepath (str), filepath (including file name) for Buy/Sell data
    budget: float, total budget in given currency
    dynamic_budget_alloc, None or float, how much of surplus profit to allocate to increase budget
    buyamount: float, "max_int", or "max", how much of a given security to buy
    sellamount: float or "all", how much of a given security to sell
    buyfee: float, fee per buy in given currency
    sellfee: float, fee per sale in given currency
    window: odd int >=3, length of filter window
    api: None or tuple, parameters for api (only relevant if act=True)
    max_loss: None or float, maximum permitted loss before selling (make sure to put a decimal point to force number to be a float)
    max_loss_absolute: bool, sets whether max_loss is absolute or relative to buy price (default=True)
    threshold_absolute: bool, sets whether threshold is absolute or relative to current price (default=True)
    act: bool, sets whether to act on a Buy/Sell signal or just add to a list
    '''
    Time = np.array(data[0])
    Price = np.array(data[1])

    BuySellData = pd.DataFrame(pd.read_csv(csv_file, header=0))
    num_owned = BuySellData.loc[len(BuySellData)-1, 'num_owned']
    surplus_profit = BuySellData.loc[len(BuySellData)-1, 'cumulative_sold'] - BuySellData.loc[len(BuySellData)-1, 'cumulative_spent']
    Buy_Prices = np.array(BuySellData['Buy_Prices'][~np.isnan(BuySellData['Buy_Prices'])])
    # Sell_Prices = np.array(BuySellData['Sell_Prices'][~np.isnan(BuySellData['Sell_Prices'])])
    if len(Buy_Prices) == 0:
        last_buy = 10*10
    else:
        last_buy = Buy_Prices[-1]

    if threshold_absolute==False:
        threshold = threshold * Price[-1]
    if max_loss_absolute==False:
        max_loss = max_loss * last_buy

    if dynamic_budget_alloc == None:
        budget_remaining = min(budget, (budget + surplus_profit))
    elif surplus_profit <= 0:
        budget_remaining = budget + surplus_profit
    elif surplus_profit > 0:
        budget_remaining = budget + dynamic_budget_alloc*surplus_profit

    if buyamount == "max_int":
        buyamount = np.floor(budget_remaining/Price[-1])
    
    elif buyamount == "max":
        buyamount = budget_remaining/Price[-1]
        
    if sellamount == "all":
        sellamount = num_owned

    ### Strategy
    df_Price = pd.DataFrame(Price)
    SMA = np.array(ta.sma(df_Price.iloc[:,0], length=window))

    check = 0 # Check to prevent multiple buys without crossing over threshold
    
    for p in range(len(Price)-window):
        i = p+window-1
        if (SMA[i] > (Price[i]+threshold)):
            check = 1
        if (SMA[i] < (Price[i]+threshold)):
            check = -1
        if (check==1) and (SMA[i] < Price[i]):
            check = 0
        if (check==-1) and (SMA[i] > Price[i]):
            check = 0

    if ((Price[-1] + buyfee) <= budget_remaining) and (check!=1) and (SMA[-1] > (Price[-1]+threshold)):
        Buy(csv_file, Price[-1], Time[-1], buyamount, buyfee, fee_absolute, api, act)

    if num_owned > 0:
        if Price[-1] <= (last_buy - max_loss):
            Sell(csv_file, Price[-1], Time[-1], sellamount, sellfee, fee_absolute, api, act)
        elif (check!=-1) and (SMA[-1] < (Price[-1]-threshold)):
            Sell(csv_file, Price[-1], Time[-1], sellamount, sellfee, fee_absolute, api, act)
    
    return

### My Strategies
def my_strategy(*args):
    pass

### Optimization
def countable_param_optimizer(data, params_to_optimize, min_list, max_list, step_list, function, other_function_params=None):
    '''
    ### Finds optimal values for parameters with countably (and finitely) many values to maximize profit for the given data
    data: depends on function, usually 2D array of format (Time, Price), data to optimize for
    params_to_optimize: 1D list, list of parameters to optimize
    min_list: 1D list, list of minimum values for parameters
    max_list: 1D list, list of maximum values for parameters
    step_list: 1D list, list of step sizes for parameters
    function: function, function to optimize for (no parameters, only function name)
    other_function_params: dict, dict of function parameters that are NOT being optimized
    # Note: this optimizes by "brute force", trying every possible combination of values,
            thus it is recommended to not optimize too many parameters at once or use very large datasets
    '''
    param_value_list = []
    for p in range(len(params_to_optimize)):
        param_value_list.append(np.arange(min_list[p], max_list[p]+step_list[p], step_list[p]))
    
    par_iterator = itertools.product(*param_value_list)
    profit_list = []
    for par in par_iterator:
        create_csv(pathmaker(""), "param_optimizer_temp.csv")
        file=pathmaker("param_optimizer_temp.csv")
        par_dict = dict(zip(params_to_optimize, par))
        function(data=data, csv_file=file, budget=10**10, *other_function_params)
        with open(file, 'r') as f:
            last_line = deque(f, 1)
            last_line = last_line.split(",")
        profit_list.append(float(last_line[-1]))
    
    profit_list = np.array(profit_list)
    max_val = np.max(profit_list)
    best_vals = np.where(profit_list == max_val)[0]

    par_list = list(par_iterator)
    best_param_list = []
    for val in best_vals:
        best_param_list.append(par_list[val])
    
    return best_param_list, params_to_optimize