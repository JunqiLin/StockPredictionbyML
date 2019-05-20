#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:40:42 2019

@author: linjunqi
"""
import numpy as np
import talib
import pandas
import time as time
import os
import quandl
from datetime import date


# Creates dataset folders in directory script is run from
try:
    os.stat("./train")
    os.stat("./eval")
except BaseException:
    os.mkdir("./train")
    os.mkdir("./eval")



barTimeframe = "1D"  # 1Min, 5Min, 15Min, 1H, 1D


assetList = ['WIKI/AAPL']


trainStartDate = date(2015,1,1)
trainEndDate = date(2017,6,1)
evalStartDate = date(2017,6,1)
evalEndDate = date(2018,6,1)

symbol = assetList[0]
return_data = quandl.get(symbol, start_date=trainStartDate, end_date=evalEndDate)

trainingDF = return_data[return_data.index < str(evalStartDate)]
evalDF = return_data[return_data.index >= str(evalStartDate)]

trainingDF.to_csv("./train/" + symbol[5:] + ".csv", index_label="date")
evalDF.to_csv("./eval/" + symbol[5:] + ".csv", index_label="date")