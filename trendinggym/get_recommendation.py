# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:26:48 2020

@author: mk
"""
# doku: http://theautomatic.net/yahoo_fin-documentation/

import os
import pickle

import requests
import pandas as pd 
from yahoo_fin import stock_info as si 
from pandas_datareader import DataReader
import numpy as np

from collections import defaultdict

from datetime import date



def get_stock_data(ticker, start, end=None):
    
    data_dir = f'data/stocks/{ticker}.csv'
    """
    #  TODO: SHOULD REALLY ONLY DOWNLOAD NEW DATA - 
    if os.path.isfile(data_dir):
        prices_cached = pd.read_csv(data_dir, index_col=0)
        start = prices_cached.index.values[-1]
    """
    stock_price = si.get_data(ticker,
                       start_date = start, 
                       end_date = end, 
                       index_as_date = True, 
                       interval = "1d")
    
    # prices_cached.merge(stock_price, how="outer")

    stock_price.to_csv(data_dir)
    
    return stock_price




today = date.today()


lhs_url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/'
rhs_url = '?formatted=true&crumb=swg7qs5y9UP&lang=en-US&region=US&' \
          'modules=upgradeDowngradeHistory,recommendationTrend,' \
          'financialData,earningsHistory,earningsTrend,industryTrend&' \
          'corsDomain=finance.yahoo.com'


tickerssp500 = si.tickers_sp500()
tickersdow = si.tickers_dow()
tickersnasdaq = si.tickers_nasdaq()

all_tickers = set(tickerssp500 + tickersdow + tickersnasdaq)

start_date = pd.Timestamp("2011-01-01")
end_date = pd.Timestamp.now()

results = {}
for ticker in all_tickers:
    results[ticker] = {}
    recommendationstrend_dict = {}
    stockinfos = []
    
    df_trend = pd.DataFrame()
              
    url =  lhs_url + ticker + rhs_url
    r = requests.get(url)
    
    if not r.ok:
        recommendation = 6
    try:
        
        # Recommendations
        result = r.json()['quoteSummary']['result'][0]
        
        recommendation = result['financialData']['recommendationMean']['fmt']
        recommendationstrend = result['recommendationTrend']['trend']
        
        m = 0
        for val in recommendationstrend:
            
            df_temp = pd.DataFrame(val, index=[today - pd.DateOffset(months = m)])
            df_temp['month'] = today.month - m
            m+=1
            
            df_trend = pd.concat([df_trend, df_temp], axis=0)
        
        #analyst info
        a_info = si.get_analysts_info(ticker)
        
        
    except KeyboardInterrupt as e:
        raise e
        
    except Exception as e:
        print(e)
        recommendation = 6
        
    results[ticker]['recommendation'] = recommendation
    results[ticker]['trend'] = df_trend
    results[ticker]['a_info'] = a_info
    
    
    
    print("--------------------------------------------")
    print ("{} has an average recommendation of: ".format(ticker), recommendation)
    #time.sleep(0.5)
    
    
    
y,m,d = end_date.year, end_date.month, end_date.day
with open(f'data/yahoo_recommendation/{y:04}-{m:02}-{d:02}.pkl', 'wb') as f:
    pickle.dump(results, f)
    

# get the stock data --> this is historic available so no rush to get it
# stock data
    
stock_data = get_stock_data("^GSPC", start_date, end_date)




























