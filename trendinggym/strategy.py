# -*- coding: utf-8 -*-
import math
import pandas as pd

def strategy(marketData,depot,memory,symbol):
    mem = []
    SMA15 = marketData["Close"].rolling(window=15).mean() #[list(marketData.keys())[0]]
    SMA50 = marketData["Close"].rolling(window=50).mean()
    if not(math.isnan(SMA15[-1]) or math.isnan(SMA50[-1]) \
           or math.isnan(SMA15[-2]) or math.isnan(SMA50[-2])):

        if (SMA15[-1]-SMA50[-1]) > 0 and (SMA15[-2]-SMA50[-2])<0:
            #buy
            buySell = 1
            numShares = math.floor((0.1*depot["cash"][-1])/(marketData["Close"][-1]))
        elif (SMA15[-1]-SMA50[-1])<0.0 and\
            (SMA15[-2]-SMA50[-2])>0.0 and not(depot["portfolio"].empty):     
            #sell
            buySell = -1
            numShares = depot["portfolio"][symbol]
        else:
            numShares = 0
            
        if numShares != 0.0:
        # create order
            orders = pd.DataFrame({"Symbol":symbol,
                      "numShares":numShares,
                      "priceStock":marketData["Close"][-1],
                      "priceOrder":numShares*marketData["Close"][-1],
                      "Date":marketData["Close"].index[-1].date(),
                      "buySell":buySell},index = [0])
        else:
            orders = pd.DataFrame({})
    else:
        orders = pd.DataFrame({})

    return orders, mem
