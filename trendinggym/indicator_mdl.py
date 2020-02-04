# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:27:11 2020

@author: mk
"""

import matplotlib.pyplot as plt
import pandas as pd
import ta
import numpy as np


filepath = "./data/test_stock.csv"
    
ticker_list = ["^GSPC"]
list_of_stocks = ticker_list
df_read = pd.read_csv(filepath, index_col=0, header=0)

# Clean NaN values
df_read = ta.utils.dropna(df_read)

def create_indicators(df_read, list_of_stocks):
    
    df_x=pd.DataFrame()
    for stock in list_of_stocks:
        
        # Extract data
        s_close = pd.Series(list(df_read[[s for s in df_read.columns if 'Close_'+stock in s and 'Adj' not in s]].values))
        s_low = pd.Series(list(df_read[[s for s in df_read.columns if 'Low_'+stock in s]].values))
        s_high = pd.Series(list(df_read[[s for s in df_read.columns if 'High_'+stock in s]].values))
        s_vol = pd.Series(list(df_read[[s for s in df_read.columns if 'Volume_'+stock in s]].values))
    
        # Calc indicators
        df_x[stock+'_macd'] = ta.trend.macd(s_close, n_slow = 26, n_fast = 12 ,fillna = False)       
        df_x[stock+'_roc'] = ta.momentum.roc(s_close, n = 12, fillna = False)        
        df_x[stock+'_wr'] = ta.momentum.wr(s_high, s_low, s_close, lbp = 14, fillna = False)        
        df_x[stock+'_mov'] = ta.volume.ease_of_movement(s_high, s_low, s_vol, n = 14, fillna = False)        
        df_x[stock+'_rsi'] = ta.momentum.rsi(s_close, n = 14, fillna = False)
        
        df_x.index = df_read.index
    
    #df = df.dropna(how='any')
    
    if False:
        plt.figure()
        plt.plot(df_read.drop(columns=[s for s in df_read.columns if "Volume" in s] ))
        
        fig, axs = plt.subplots(5)
        fig.suptitle('Indicators')
        axs[0].plot(df_read.index, df_x[stock+'_rsi'].values)
        axs[1].plot(df_read.index, df_x[stock+'_mov'].values)
        axs[2].plot(df_read.index, df_x[stock+'_wr'].values)
        axs[3].plot(df_read.index, df_x[stock+'_roc'].values)
        axs[4].plot(df_read.index, df_x[stock+'_macd'].values)
  

    # forecast = 7: t0 = first index, t_end = real index - 7 - 
    df_y = pd.DataFrame()
    
    df_y['+7'] = np.concatenate(s_close[7:].values)
    df_y['+6'] = np.concatenate(s_close[6:-1].values)
    df_y['+5'] = np.concatenate(s_close[5:-2].values)
    df_y['+4'] = np.concatenate(s_close[4:-3].values)
    df_y['+3'] = np.concatenate(s_close[3:-4].values)
    df_y['+2'] = np.concatenate(s_close[2:-5].values)
    df_y['+1'] = np.concatenate(s_close[1:-6].values)
    
    df_y.index = df_read.index[:-7]
    
    # align index and clean data 

    
    # save training data
        
if __name__ == '__main__':
	create_indicators(df_read, ticker_list)
    
    
    
    
    
    
    
    
    
    