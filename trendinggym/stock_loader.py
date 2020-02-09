# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:12:45 2020

@author: mk
"""


# https://pandas-datareader.readthedocs.io/en/latest/index.html

import os
import pandas as pd

from pandas_datareader import data


def main():
    
    filepath = "./data/stock_chart_data.csv"
    
    ticker_list = ["^GSPC"]
    
    start_date = pd.Timestamp("2011-01-01")
    end_date = pd.Timestamp.now()
    
    
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath, index_col=0, header=0)
        df.index = pd.to_datetime(df.index)
        start_date = max(max(df.index), start_date)
        start_date = start_date + pd.Timedelta("1d")
    else:
        df = pd.DataFrame([])
        
    if start_date > end_date:
        raise ValueError('The start date is in the future - no new stock data saved')
        
    # Read historical stock data
    df_stock_data = data.DataReader(ticker_list, 'yahoo', start_date, end_date)
    
    # Rename multi index columns
    df_stock_data.columns = [col+'_'+ticker for ticker in ticker_list 
                             for col in list(df_stock_data.columns.get_level_values(0))]

    # Append new stock data
    result = df.append(df_stock_data, sort=False)
    
    result.to_csv(filepath)
        
    result.drop(columns=[s for s in df_stock_data.columns if "Volume" in s] ).plot()
    
	
if __name__ == '__main__':
	main()
