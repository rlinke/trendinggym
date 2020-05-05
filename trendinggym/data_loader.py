
# https://www.researchgate.net/publication/334306834_Forecasting_stock_market_movements_using_Google_Trend_searches

import os
import pandas as pd

from pytrends.request import TrendReq
import itertools    
from fp.fp import FreeProxy


def get_bookkeeper(cache_path, elements):    
    start_date = pd.Timestamp("2018-01-01")
    end_date = pd.Timestamp.now()
    freq = "1h"
    
    dates = list(pd.date_range(start_date, end_date, freq=freq))
    
    if os.path.isfile(cache_path):
        bookkeeper = pd.read_pickle(cache_path)
        print("loaded cached bookkeeper with: {0} elements".format(bookkeeper.notna().sum().sum()))
    else:
        bookkeeper = pd.DataFrame(data=None, index=elements, columns=dates)
        bookkeeper.index.name = "keyword"

    return bookkeeper

def update_bookkeeper(df):
    global bookkeeper
    bookkeeper.loc[df.columns,df.index.values] = df.transpose()


def check_range(start_date, end_date, elements):
    global bookkeeper
    
    to_do = []
    for element in elements:
        # element = elements[0]
        if bookkeeper.loc[element, start_date:end_date].isna().sum():
            to_do.append(element)
    return to_do


def do_request(ptr, chunk, start_date, end_date):

    print("doing google request")
    df_new = ptr.get_historical_interest(chunk, 
                year_start=start_date.year, 
                month_start=start_date.month, 
                day_start=start_date.day, 
                hour_start=start_date.hour, 
                year_end=end_date.year,
                month_end=end_date.month, 
                day_end=end_date.day, 
                hour_end=end_date.hour, 
                cat=0, 
                geo='', 
                gprop='', 
                sleep=60
               )
    
    print("finished request")
    if 'isPartial' in df_new.columns:
        df_new = df_new.drop('isPartial', axis=1)
        
    return df_new



def main(elements):

    cache_path = "data/test.pkl"
    
    bookkeeper = get_bookkeeper(cache_path, elements)
    
    
    ptr = TrendReq(hl='en-US',
                   tz=360, 
                   retries =5,
                   backoff_factor =1.,
                   timeout=10.,
                   proxies=['https://'+i for i in FreeProxy().get_proxy_list()])

    print("starting with {0} to {1}".format(start_date, end_date))
    weeks = list(pd.date_range(start_date,end_date, freq="1w"))
    ranges = list(zip(weeks[0:-1],weeks[1:]))
    
    for start_date, end_date in ranges:
        #start_date, end_date = ranges[0]
        print(start_date, end_date)
        # check if data already filled:
        elements_to_do = check_range(start_date, end_date, elements)
        print(len(elements_to_do))
        if len(elements_to_do):
            print("checked range it will be handled!")
            element_chunks = [elements_to_do[i:i+5] for i in range(0, len(elements_to_do), 5)]
            for chunk in element_chunks:
                chunk = element_chunks[0]
                print("reading {0}".format(chunk))
                df = do_request(ptr, chunk, start_date, end_date)
                
                if len(df):
                    update_bookkeeper(df)
                
                    bookkeeper.to_pickle(cache_path)

        else:
            print("skipping range")

"""
start_date = pd.Timestamp("2020-03-31")
end_date = pd.Timestamp("2020-04-30")
"""
def main():
    filepath = "data/test.csv"
    
    trending_topics = ["economy", "energy", "bonds", "crisis", "finance"]
    
    start_date = pd.Timestamp("2011-01-01")
    end_date = pd.Timestamp.now()
    
    if os.path.isfile(filepath):
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index)
        start_date = max(max(df.index), start_date)
        start_date = start_date + pd.Timedelta("1h")
    else:
        df = pd.DataFrame([])
    
    print("starting with {0} to {1}".format(start_date, end_date))
    
    ptr = TrendReq(hl='en-US', tz=1)
    
    months = list(pd.date_range(start_date,end_date, freq="1m"))
    
    ranges = list(zip(months[0:-1],months[1:]))
    date_range = ranges
    for start_date, end_date in ranges:
        print(start_date, end_date)
        df_new = ptr.get_historical_interest(trending_topics, 
                                         year_start=start_date.year, 
                                         month_start=start_date.month, 
                                         day_start=start_date.day, 
                                         hour_start=start_date.hour, 
                                         year_end=end_date.year,
                                         month_end=end_date.month, 
                                         day_end=end_date.day, 
                                         hour_end=end_date.hour, 
                                         cat=0, 
                                         geo='', 
                                         gprop='', 
                                         sleep=60
                                        )
        
    
        df_sub = df_new.drop('isPartial', axis=1)
        
        if len(df_sub):
            df = pd.concat([df, df_new])    
            df.to_csv(filepath)
    
    df2.plot()
	
if __name__ == '__main__':
	main()

	elements = 	"""economy 
        energy
        bonds 
        crisis
        finance
        growth 
        stocks 
        conflict
        derivatives
        culture
        investment
        revenue
        short.selling
        hedge
        profit
        inflation
        portfolio
        bubble
        consume
        money
        returns
        society
        banking
        environment
        dow.jones
        nasdaqs
        tock.market
        unemployment
        economics
        return
        risk
        markets
        dividend
        metals
        leverage
        loss
        religion
        consumption
        transaction
        politics
        sp500_ts
        housing
        tourism
        war
        earnings
        chance
        cash
        arts
        default
        invest
        oil
        fond
        house
        present
        fed
        fun
        gains
        forex
        credit
        garden
        rare.earths
        success
        travel
        office
        cancer
        headlines
        kitchen
        car
        water
        sell
        debt
        colorring
        restaurant
        freedom
        buy
        happy
        rich
        crash
        gain
        gold
        world
        lifestyle
        fine
        trader
        home
        labor
        holiday
        marriage
        train"""
        
    elements = elements.split('\n')
    elements = [e.strip() for e in neff]
    
