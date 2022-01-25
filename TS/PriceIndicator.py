# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:59:30 2020

@author: Somnath
"""

def GetPriceFrequency(PriceData):
    frequency = "D"
    for col in PriceData.columns: 
        if("Close Price" in col):
            frequency = col[-1]
            break
    return frequency

def GetPriceRange(PriceData):
    frequency = GetPriceFrequency(PriceData)
            
    PriceData['Range'] = PriceData["High Price" + "_" + frequency] - PriceData["Low Price" + "_" + frequency]
    return PriceData

def GetPriceMovementPerRecord(PriceData):
    frequency = GetPriceFrequency(PriceData)
            
    PriceData['PercentMovement'] = (PriceData["High Price" + "_" + frequency] / PriceData["Low Price" + "_" + frequency]-1) * 100.00
    return PriceData

def GetMidPrice(PriceData):
    frequency = GetPriceFrequency(PriceData)
            
    PriceData['MidPrice'] = (PriceData["High Price" + "_" + frequency] + PriceData["Low Price" + "_" + frequency])/2
    return PriceData

# CloseLine
# MoneyFlow
# RSI
# Stochastic