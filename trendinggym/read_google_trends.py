# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:21:45 2020

@author: mk
"""


from pytrends.request import TrendReq
from pytrends import dailydata
import numpy as np
import pickle
import time
import os
import pandas as pd
import matplotlib.pyplot as plt



def collect_trend_data():
    
    
    years =  list(np.arange(2010, 2020, 1))
    months = list(np.arange(1, 12, 1))
    
    
    df_dict = {}
    stock_dict = {}
    words = ["cash","bubble","return","stocks","gain","transaction",
                "dividend","revenue","stock market","debt"]
    #words = ["transaction"]
    
    
    for word in words:
        
        df_dict = {}
        for y in years:    
            for m in months:
                print("going to process word -{}- for {}-{}".format(word, y, m+1))
                time.sleep(10)
                df_dict[str(y)+str(m+1)] = dailydata.get_daily_data(word, y, m, y, m+1, geo = '')
                print("processed done")
    
        time.sleep(10)           
        with open("./trend/"+word+".pickle", 'wb') as handle:
            pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                 
        stock_dict[word] = df_dict
    
    
    return df_dict, stock_dict
    


def tranform_trend_data():
    
    trend_path = "./trend"
    dict_load = {}
    for item in os.listdir(trend_path):
    
        with open(os.path.join(trend_path, item), 'rb') as handle:
            read_data = pickle.load(handle)
            
            
        df_trend = pd.concat(read_data)
        df_trend = df_trend.reset_index(level=[0], drop=True).sort_index()
        
        dict_load[item] = df_trend.loc[~df_trend.index.duplicated(keep='first')]
                
            
    with open("./trend/raw_trend.pickle", 'wb') as handle:
        pickle.dump(dict_load, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


    return dict_load


if __name__ == '__main__':
    
    
    df_dict, stock_dict = collect_trend_data()
    
    dict_load = tranform_trend_data()











