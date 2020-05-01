# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:12:45 2020

@author: mk
"""
# https://pandas-datareader.readthedocs.io/en/latest/index.html

import os
import pandas as pd
import pickle
from pandas_datareader import data


def main():
    
    path = './data'
    file = 'stock_chart_data.pickle'
    checkpoint = 'checkpoint.pickle'
    
    ticker_list = ["^GSPC"]
    
    start_date = pd.Timestamp("2011-01-01")
    end_date = pd.Timestamp.now()
    
    dict_save = {}
    
    for ticker in ticker_list:
             
        if os.path.isfile(os.path.join(path, file)) and os.path.isfile(os.path.join(path, checkpoint)):
            
            with open(os.path.join(path, checkpoint), 'rb') as handle:
                start_date = pickle.load(handle)
            
            start_date = start_date + pd.Timedelta("1d")
            
            with open(os.path.join(path, file), 'rb') as handle:
                df = pickle.load(handle)
            
        else:
            df = pd.DataFrame([])
            
        if start_date > end_date:
            raise ValueError('The start date is in the future - no new stock data saved')
            
        # Read historical stock data
        df_stock_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
        
        # Rename multi index columns
        df_stock_data.columns = [col+'_'+ticker for ticker in ticker_list 
                                 for col in list(df_stock_data.columns.get_level_values(0))]
    
        # Append new stock data
        result = df.append(df_stock_data, sort=False)
        
        dict_save[ticker] = result
    
    
    # save stock data 
    with open(os.path.join(path, file), 'wb') as handle:
        pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # save checkpoint
    with open(os.path.join(path,checkpoint), 'wb') as handle:
        pickle.dump(end_date, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    #result.drop(columns=[s for s in df_stock_data.columns if "Volume" in s] ).plot()
    
	
if __name__ == '__main__':
	main()
