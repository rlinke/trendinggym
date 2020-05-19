# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:41:38 2020

@author: Richard
"""

import pandas as pd
import numpy as np
import pickle


class Orderbook:
    def __init__(self, portfolio, order_cost, df):
        self.df = df
        self.portfolio = portfolio
        self.order_cost = order_cost
        self.history = []
        self.loc = {
            "type": 0,
            "price": 1,
            "amount": 2,
            "ts": 3,
            "current_value": 4
            }
        self.last_order = [None,0,0,0,0]
        
    def sell_option(self, ts_now):
        if self.last_order[self.loc["type"]] == None:
            return
        
        elif self.last_order[self.loc["type"]] == "long":
            current_price = self.df.loc[ts_now, "close"]
            number_options = self.last_order[self.loc["amount"]]
            self.portfolio = max(current_price * number_options - self.order_cost, 0)
            
        elif self.last_order[self.loc["type"]] == "short":
            current_price = self.df.loc[ts_now, "close"]
            original_price = self.last_order[self.loc["price"]]
            number_options = self.last_order[self.loc["amount"]]
            residual = (original_price - current_price)
            self.portfolio = max((original_price + residual) * number_options - self.order_cost, 0)
              
        
    def buy(self, ts, short=False):
        keep_option = False
        if (short == True and self.last_order[self.loc["type"]] == "short") \
            or (short == False and self.last_order[self.loc["type"]] == "long"):
            keep_option = True
          
        if not keep_option:
            self.sell_option(ts)
        
        current_price = self.df.loc[ts, "close"]
        
        if not keep_option:
            self.portfolio -= self.order_cost            
            orders = self.portfolio / current_price
            self.last_order = ["long", current_price, orders, ts, orders * current_price] 
        
            if short:
                self.last_order[0] = "short"

        else:
            self.last_order =[self.last_order[0], 
                              current_price, 
                              self.last_order[2],
                              self.last_order[3],
                              self.last_order[2] * current_price]
        self.portfolio = 0
        self.history.append(self.last_order)
       
#%%

#if __name__ == '__main__':

filepath = "data/stocks/^GSPC.csv"
df = pd.read_csv(filepath, index_col=0)
df.index = pd.to_datetime(df.index.values)
df.drop(['open', 'high', 'low', 'adjclose', 'volume', 'ticker'], axis=1, inplace=True)




buy, hold, sell = [], [] , []

order_cost = 10
portfolio = 10000
history = []


baseline = pd.read_pickle("data/cache/2020_05_13_baseline.pkl")

df["predicted"] = baseline["predicted"]

df.dropna(how="any", inplace=True)

ob = Orderbook(portfolio, order_cost, df)

for i in range(len(df)):
    ts = df.index[i]
    if df["predicted"].iloc[i] == 2:
        ob.buy(ts)
        
    elif df["predicted"].iloc[i] == 1:
       ob.buy(ts, True)

# after last just sell to the last closing price
ob.sell_option(ts)
print(ob.portfolio)    


results = ob.history

res = pd.DataFrame(results, columns=ob.loc.keys())
res.index = res["ts"]
res.drop("ts", axis=1, inplace=True)
res["current_value"].plot()








