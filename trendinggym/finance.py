# -*- coding: utf-8 -*-
from strategy import strategy
import pandas as pd
import datetime, time
import numpy as np
import math
from datascraper import getYahooData
from backtest import backtest

ticker_list = ['AAPL']

marketData = {'AAPL': getYahooData('AAPL',datetime.datetime(2002,1,1),datetime.datetime.now(),'1')}

depot ={"cash":np.array([10000],dtype = float),
        "orders":pd.DataFrame({"Symbol":[],
                  "numShares":[],
                  "priceStock":[],
                  "priceOrder":np.array([]),
                  "Date":[],
                  "buySell":[]}),
        "portfolio": pd.DataFrame({"Symbol":[],
                      "numShares":[]}),
        "fOrderCost": lambda x: min(max(x*0.0025,10.0),75.0)}
trader = {"memory":[],
          "fStrategy": strategy}

depot = backtest(marketData,depot,trader)

