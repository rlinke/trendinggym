# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:26:48 2020

@author: mk
"""
# doku: http://theautomatic.net/yahoo_fin-documentation/

import requests
import pandas as pd 
from yahoo_fin import stock_info as si 
from pandas_datareader import DataReader
import numpy as np

from collections import defaultdict

from datetime import date

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

recommendations = []
recommendationstrend_dict = {}


for ticker in tickersdow:

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
        
        # stock data
        stock_price = get_data(ticker, start_
                               date = start_date, 
                               end_date = end_date, 
                               index_as_date = True, interval = “1d”)
        
        
    except:
        recommendation = 6
    
    recommendations.append(recommendation)
    
    recommendationstrend_dict[ticker] = df_trend
    
    
    
    print("--------------------------------------------")
    print ("{} has an average recommendation of: ".format(ticker), recommendation)
    #time.sleep(0.5)
    
    
    
    
#dataframe = pd.DataFrame(list(zip(tickers, recommendations)), columns =['Company', 'Recommendations']) 
#dataframe = dataframe.set_index('Company')
#dataframe.to_csv('recommendations.csv')

#print (df)