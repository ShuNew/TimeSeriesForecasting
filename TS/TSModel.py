# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 22:56:46 2022

@author: Somnath
"""
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt

# Decomposition
str_multiplicative = "multiplicative"
str_additive = "additive"

def decompose(data,modelindicator,period,showplot):
    decomposeresult = seasonal_decompose(data, model=modelindicator, period=period)

    if(showplot):
        decomposeresult.plot()
        plt.show()
    
    return decomposeresult