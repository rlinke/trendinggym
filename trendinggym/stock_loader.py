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
    
    filepath = "/data/test_stock.csv"
    
    ticker_list = ["^GSPC"]
    
    start_date = pd.Timestamp("2011-01-01")
    end_date = pd.Timestamp.now()
    
    
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath, index_col=0, header=[0,1])
        df.index = pd.to_datetime(df.index)
        start_date = max(max(df.index), start_date)
        start_date = start_date + pd.Timedelta("1d")
    else:
        df = pd.DataFrame([])
        
    if start_date > end_date:
        raise ValueError('The start date is in the future - no new stock data saved')
        
    # Read historical stock data
    df_stock_data = data.DataReader(ticker_list, 'yahoo', start_date, end_date)

    # Append new stock data
    result = df.append(df_stock_data, sort=False)
    
    result.to_csv(filepath)
        
    result.plot()
	
if __name__ == '__main__':
	main()
