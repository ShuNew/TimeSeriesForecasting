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

def getData(query_switch, function, symbol, interval, *optionalparams):
    '''
    ### Get data using AlphaVantage API
    # requires an API key in a text file "AlphaVantage_apikey.txt" located in the same directory as "AlphaVantage.py"
    query_switch: bool, if True then query AlphaVantage and save .csv file, if False then read saved .csv file
    function, symbol, interval, *optionalparams: see AlphaVantage documentation
    '''
    file_name = str(function)+'-'+str(symbol)+'-'+str(interval)+'.csv'
    fullfilename = pathmaker(file_name)
    if query_switch == True:
        if optionalparams:
            url = 'https://www.alphavantage.co/query?'+'function='+str(function)+'&'+'symbol='+str(symbol)+'&'+'interval='+str(interval)+'&'+'apikey='+str(apikey)+'&datatype=csv'+str(optionalparams)
        else:
            url = 'https://www.alphavantage.co/query?'+'function='+str(function)+'&'+'symbol='+str(symbol)+'&'+'interval='+str(interval)+'&'+'apikey='+str(apikey)+'&datatype=csv'
        data = pd.read_csv(url)
        data.to_csv(fullfilename, index=False)
    if query_switch == False:
        data = pd.DataFrame(pd.read_csv(fullfilename))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by="timestamp")
    data = data.reset_index(drop=True)
    return(data)
