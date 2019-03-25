#from datetime import datetime
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


targetLookaheadPeriod = 1
startCutoffPeriod = 50  # Set to length of maximum period indicator


# Tracks position in list of symbols to download
iteratorPos = 0
assetListLen = len(assetList)


while iteratorPos < assetListLen:
#while False:
#    try:
        symbol = assetList[iteratorPos]
        returned_data = quandl.get(symbol, start_date=trainStartDate, end_date=evalEndDate)
        
        # Processes all data into numpy arrays for use by talib
        timeList = np.array(returned_data.index)
        openList = np.array(returned_data['Open'], dtype=np.float64)
        highList = np.array(returned_data['High'], dtype=np.float64)
        lowList = np.array(returned_data['Low'], dtype=np.float64)
        closeList = np.array(returned_data['Close'], dtype=np.float64)
        volumeList = np.array(returned_data['Volume'], dtype=np.float64)

        # Adjusts data lists due to the reward function look ahead period
        shiftedTimeList = timeList[:-targetLookaheadPeriod]
        shiftedClose = closeList[targetLookaheadPeriod:]
        highList = highList[:-targetLookaheadPeriod]
        lowList = lowList[:-targetLookaheadPeriod]
        closeList = closeList[:-targetLookaheadPeriod]

        # Calculate trading indicators
        RSI14 = talib.RSI(closeList, 14)
        RSI50 = talib.RSI(closeList, 50)
        STOCH14K, STOCH14D = talib.STOCH(
            highList, lowList, closeList, fastk_period=14, slowk_period=3, slowd_period=3)

        # Calulate network target/ reward function for training
        closeDifference = shiftedClose - closeList
        closeDifferenceLen = len(closeDifference)

        # Creates a binary output if the market moves up or down, for use as
        # one-hot labels
        longOutput = np.zeros(closeDifferenceLen)
        longOutput[closeDifference >= 0] = 1
        shortOutput = np.zeros(closeDifferenceLen)
        shortOutput[closeDifference < 0] = 1

        # Constructs the dataframe and writes to CSV file
        outputDF = {
            "close": closeList,  # Not to be included in network training, only for later analysis
            "RSI14": RSI14,
            "RSI50": RSI50,
            "STOCH14K": STOCH14K,
            "STOCH14D": STOCH14D,
            "longOutput": longOutput,
            "shortOutput": shortOutput
        }
        # Makes sure the dataframe columns don't get mixed up
        columnOrder = ["close", "RSI14", "RSI50", "STOCH14K",
                       "STOCH14D", "longOutput", "shortOutput"]
        outputDF = pandas.DataFrame(
            data=outputDF,
            index=shiftedTimeList,
            columns=columnOrder)[
            startCutoffPeriod:]
        
         #Splits data into training and evaluation sets
        trainingDF = outputDF[outputDF.index < str(evalStartDate)]
        evalDF = outputDF[outputDF.index >= str(evalStartDate)]

        if (len(trainingDF) > 0 and len(evalDF) > 0):
#            print("writing " + str(symbol) +
#                  ", data len: " + str(len(closeList)))

            trainingDF.to_csv("./train/" + symbol[5:] + ".csv", index_label="date")
            evalDF.to_csv("./eval/" + symbol[5:] + ".csv", index_label="date")
            

