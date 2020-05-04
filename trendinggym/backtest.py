# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def backtest(marketData,depot,trader):
    
    symbol = list(marketData.keys())[0]
    day_start = marketData[symbol].index[0].date()
    for day_current, day_data in marketData[symbol].iterrows():  
        if day_current >= day_start: # catch if current day are the same or current day is too far in history
            [orders,memStgy] = trader["fStrategy"](marketData[symbol][day_start:day_current.date()]\
                                                   ,depot,trader["memory"],symbol)#

            for index, orderData in orders.iterrows(): # in case the trader goes for multiple orders
                depot["orders"] = depot["orders"].append(orderData)
                depot["portfolio"] = updatePortfolio(depot)
                
                orderCost = depot["fOrderCost"](orderData["priceOrder"])
                depot["cash"] = np.append(depot["cash"],depot["cash"][-1]\
                                     -orderCost\
                                     -orderData["buySell"]*orderData["priceOrder"])
                if depot["cash"][-1] <= 0.0:
                    print("You ran out of money! I won't give you some.")
                    return
         #maybe add some plotting here
        
    return depot
    
                
                
def updatePortfolio(depot):
    fBuySellSum = lambda x: sum(np.multiply(x["numShares"],x["buySell"]))
    
    orders = depot["orders"]
    dfGroupObj = orders[["Symbol","numShares","buySell"]].groupby(["Symbol"])
    portfolio = dfGroupObj.apply(fBuySellSum)
    portfolio.name = "numShares" # maybe dependend on dict of depot (but found no clever way)
    #portfolio.reset_index()
    portfolio.drop(portfolio[portfolio.values == 0.0].index,inplace=True)
    return portfolio


        
