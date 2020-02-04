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

df_read = pd.read_csv(filepath, index_col=0, header=0)

# Clean NaN values
df_read = ta.utils.dropna(df_read)

def create_indicators(df_read, list_of_stocks):
    
    df=pd.DataFrame()
    for stock in list_of_stocks:
        
        # Extract data
        s_close = pd.Series(list(df_read[[s for s in df_read.columns if 'Close_'+stock in s and 'Adj' not in s]].values))
        s_low = pd.Series(list(df_read[[s for s in df_read.columns if 'Low_'+stock in s]].values))
        s_high = pd.Series(list(df_read[[s for s in df_read.columns if 'High_'+stock in s]].values))
        s_vol = pd.Series(list(df_read[[s for s in df_read.columns if 'Volume_'+stock in s]].values))
    
        # Calc indicators
        df[stock+'_macd'] = ta.trend.macd(s_close, n_slow = 26, n_fast = 12 ,fillna = True)       
        df[stock+'_roc'] = ta.momentum.roc(s_close, n = 12, fillna = True)        
        df[stock+'_wr'] = ta.momentum.wr(s_high, s_low, s_close, lbp = 14, fillna = True)        
        df[stock+'_mov'] = ta.volume.ease_of_movement(s_high, s_low, s_vol, n = 14, fillna = True)        
        df[stock+'_rsi'] = ta.momentum.rsi(s_close, n = 14, fillna = True)
        
    
    df = df.dropna(how='any')
    
    if False:
        plt.figure()
        plt.plot(df_read.drop(columns=[s for s in df_read.columns if "Volume" in s] ))
        
        fig, axs = plt.subplots(5)
        fig.suptitle('Indicators')
        axs[0].plot(df_read.index, df[stock+'_rsi'].values)
        axs[1].plot(df_read.index, df[stock+'_mov'].values)
        axs[2].plot(df_read.index, df[stock+'_wr'].values)
        axs[3].plot(df_read.index, df[stock+'_roc'].values)
        axs[4].plot(df_read.index, df[stock+'_macd'].values)
  


forecast = 7

df_y = pd.DataFrame(columns=['+7','+6','+5','+4','+3','+2','+1'])

df_y = pd.DataFrame()

df_y['+7'] = s_close[7:]
df_y['+6'] = s_close[6:]
df_y['+5'] = s_close[5:]
df_y['+4'] = s_close[4:]
df_y['+3'] = s_close[3:]
df_y['+2'] = s_close[2:]
df_y['+1'] = s_close[1:]

#for index in range(0,len(s_close)):
#
#    if index == len(s_close)-forecast:
#        break
#    
#    df_y.insert(index, '+7', s_close[index+7][0])  
#    
#    print(s_close[index+7][0])

    
    
        
if __name__ == '__main__':
	create_indicators(df_read, ticker_list)
    
    
    
    
    
    
    
    
    
    