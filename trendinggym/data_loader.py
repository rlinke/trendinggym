
# https://www.researchgate.net/publication/334306834_Forecasting_stock_market_movements_using_Google_Trend_searches

import os
import pandas as pd

from pytrends.request import TrendReq


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
    
    ptr = TrendReq(hl='en-US', tz=1)
    
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
    
    df.to_csv(filepath)
        
    df.plot()
	
if __name__ == '__main__':
	main()

	
	"""
	M. Y. Huang et economy energy bonds crisis finance growth stocks conflictderivativescultureinvestmentrevenueshort.sellinghedgeprofitinflationportfoliobubbleconsumemoneyreturnssocietybankingenvironmentdow.jonesnasdaqstock.marketunemploymenteconomicsreturnriskmarketsdividendmetalsleveragelossreligionconsumptiontransactionpoliticssp500_tshousingtourismwarearningschancecashartsdefaultinvestoilfondhousepresentfedfungainsforexcreditgardenrare.earthssuccesstravelofficecancerheadlineskitchencarwaterselldebtcolorringrestaurantfreedombuyhappyrichcrashgaingoldworldlifestylefinetraderhomelaborholidaymarriagetrain
	"""