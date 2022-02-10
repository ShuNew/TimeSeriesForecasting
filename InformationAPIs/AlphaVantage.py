import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# switch = True

# function = "TIME_SERIES_INTRADAY"
# symbol = "IBM"
# interval = "5min"
# optionalparams = ""

def pathmaker(filename):
    file_name = filename
    fullfilename = os.path.join(os.path.dirname(__file__), file_name)
    return(fullfilename)

api_file = pathmaker("AlphaVantage_apikey.txt")
apikey = np.loadtxt(api_file, dtype=str)

def getData(query_switch, function, symbol, *optionalparams):
    '''
    ### Get data using AlphaVantage API
    # requires an API key in a text file "AlphaVantage_apikey.txt" located in the same directory as "AlphaVantage.py"
    query_switch: bool, if True then query AlphaVantage and save .csv file, if False then read saved .csv file
    (function, symbol, interval, *optionalparams): see AlphaVantage documentation
    '''
    if optionalparams:
        op_name = ''
        for param in optionalparams:
            op_name += '-' + str(param)
        file_name = str(function)+'-'+str(symbol)+op_name+'.csv'
    else:
        file_name = str(function)+'-'+str(symbol)+'.csv'
    fullfilename = pathmaker(file_name)
    if query_switch == True:
        if optionalparams:
            op_param = ''
            for param in optionalparams:
                op_param += '&' + str(param)
            url = 'https://www.alphavantage.co/query?'+'function='+str(function)+'&'+'symbol='+str(symbol)+op_param+'&'+'apikey='+str(apikey)+'&datatype=csv'
        else:
            url = 'https://www.alphavantage.co/query?'+'function='+str(function)+'&'+'symbol='+str(symbol)+'&'+'apikey='+str(apikey)+'&datatype=csv'
        data = pd.read_csv(url)
        data.to_csv(fullfilename, index=False)
    if query_switch == False:
        data = pd.DataFrame(pd.read_csv(fullfilename))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by="timestamp")
    data = data.reset_index(drop=True)
    return(data)

def getDataGeneral(query_switch, file, function, symbol, *optionalparams):
    '''
    ### Get data using AlphaVantage API
    # requires an API key in a text file "AlphaVantage_apikey.txt" located in the same directory as "AlphaVantage.py"
    query_switch: bool, if True then query AlphaVantage and save .csv file, if False then read saved .csv file
    file: string (filepath), file path (including file name)
    (function, symbol, interval, *optionalparams): see AlphaVantage documentation
    '''
    fullfilename = file
    if query_switch == True:
        if optionalparams:
            url = 'https://www.alphavantage.co/query?'+'function='+str(function)+'&'+'symbol='+str(symbol)+str(optionalparams[0])+'&'+'apikey='+str(apikey)+'&datatype=csv'
        else:
            url = 'https://www.alphavantage.co/query?'+'function='+str(function)+'&'+'symbol='+str(symbol)+'&'+'apikey='+str(apikey)+'&datatype=csv'
        data = pd.read_csv(url)
        data.to_csv(fullfilename, index=False)
    if query_switch == False:
        data = pd.DataFrame(pd.read_csv(fullfilename))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by="timestamp")
    data = data.reset_index(drop=True)
    return(data)