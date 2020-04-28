# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:27:11 2020

@author: mk
"""
import pandas as pd
import ta
import numpy as np

filepath = "./data/stock_chart_data.csv"
save_path = "./data/stock_feature_data.csv"
    
ticker_list = ["^GSPC"]

df_read = pd.read_csv(filepath, index_col=0, header=0)

# Clean NaN values
df_read = ta.utils.dropna(df_read)

def create_indicators(df_read, list_of_stocks, path):
    
    df_x=pd.DataFrame()
    param = 12
    
    for stock in list_of_stocks:
        
        # Extract data - https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html
        s_close = pd.Series(df_read[[s for s in df_read.columns if 'Close_'+stock in s and 'Adj' not in s]].values[:,0])
        s_low = pd.Series(df_read[[s for s in df_read.columns if 'Low_'+stock in s]].values[:,0])
        s_high = pd.Series(df_read[[s for s in df_read.columns if 'High_'+stock in s]].values[:,0])
        s_vol = pd.Series(df_read[[s for s in df_read.columns if 'Volume_'+stock in s]].values[:,0])
    
        # Calc indicators
        df_x[stock+'_macd'] = ta.trend.macd(s_close, n_slow = 26, n_fast = param ,fillna = False)       
        df_x[stock+'_roc'] = ta.momentum.roc(s_close, n = 12, fillna = False)   
        df_x[stock+'_wr'] = ta.momentum.wr(s_high, s_low, s_close, lbp = 14, fillna = False)       
        df_x[stock+'_mov'] = ta.volume.ease_of_movement(s_high, s_low, s_vol, n = 14, fillna = False)        
        df_x[stock+'_rsi'] = ta.momentum.rsi(s_close, n = 14, fillna = False)
        df_x[stock+'_close'] = s_close.values

        
        df_x.index = df_read.index  

    # forecast = 7: t0 = first index, t_end = real index - 7 - 
    df_y = pd.DataFrame()
    
    # Better use shift function !!!
    df_y['+7'] = s_close[7:].values
    df_y['+6'] = s_close[6:-1].values
    df_y['+5'] = s_close[5:-2].values
    df_y['+4'] = s_close[4:-3].values
    df_y['+3'] = s_close[3:-4].values
    df_y['+2'] = s_close[2:-5].values
    df_y['+1'] = s_close[1:-6].values
    df_y['0'] = s_close[:-7].values
    df_y['diff'] = df_y['0'] - df_y['+1']
        
    df_y.index = df_read.index[:-7]
    
    # align index and clean data   
    df_data_save = pd.concat([df_x, df_y.reindex(df_x.index)], axis=1)
    df_data_save = df_data_save.tail(len(df_data_save)-param-1)
    
    # save training data
    df_data_save.to_csv(path)
     
    return df_data_save
        
if __name__ == '__main__':
    
	create_indicators(df_read, ticker_list, save_path)
    
    
    
    
    
    
    
    
    
    