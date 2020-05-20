# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:36:19 2020

@author: mk
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from pandas.tseries.offsets import BDay
from collections import defaultdict

from trading_utils import load_feature_data, combine_swing, stopp_strategy

#from keras.models import load_model
from tensorflow.keras.models import load_model

from predict_mdl import preprocessing, prepare_data, create_train_mdl
from predict_mdl import *

import logging
import logging.handlers
import os

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
 
handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "./LOG_portfolio.log"))
formatter = logging.Formatter(logging.BASIC_FORMAT)
handler.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
root.addHandler(handler)
 
logging.getLogger().setLevel(logging.INFO)



''' Define global Timeline and Parameters'''

start_date = pd.Timestamp("2010-01-01")

end_train_date = pd.Timestamp("2017-01-01")

end_date = pd.Timestamp("2018-01-01")

order_cost =  5

money = 10000

''' ---------------------------------------------------------  '''



def calc_baseline_profit(df, ticker):
    '''
    Calculate basline balance and profit

    Parameters
    ----------
    df : dataframe
        Stock price
    ticker : string
        Which ticker (= stock) should be calcuted

    Returns
    -------
    profit : float64
        profit in per cent 
    balance_end : float64
        Balance (depot value)
    '''
    
    ticker_close = ticker + '_close'
    
    balance_init = money - order_cost
    
    stock_init = balance_init/df[ticker_close][0]
    balance_end = stock_init * df[ticker_close][-1]
    
    profit = ((balance_end - order_cost)/balance_init ) * 100
    
    return profit, balance_end
    

def init_train(df, flag):
    '''
    Initial training

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    init_mdl : TYPE
    df_x_train : TYPE
    df_y_train : TYPE

    '''

    df_x_train, df_y_train, df_y_swing = preprocessing(df)
    X, Y_en, Y_s = prepare_data(df_x_train, df_y_train, df_y_swing)
        
    if flag:
        
        init_mdl_lstm, init_mdl_tcn = create_train_mdl(X, Y_en, Y_s)
        
        print('Trained and saved init model')
        
        # save training data
        with open('./data/init_mdl_lstm.pickle', 'wb') as handle:
            pickle.dump(init_mdl_lstm, handle, protocol=pickle.HIGHEST_PROTOCOL)

        init_mdl_tcn.save('./data/init_mdl_tcn.h5')
        
        # with open('./data/init_mdl_tcn.pickle', 'wb') as handle:
        #      pickle.dump(init_mdl_tcn, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    else:
        
        print('Load init models ')
        with open('./data/init_mdl_lstm.pickle', 'rb') as handle:
            init_mdl_lstm = pickle.load(handle)

        init_mdl_tcn = load_model('./data/init_mdl_tcn.h5', custom_objects={'TCN': TCN}) 
        
    return init_mdl_lstm, init_mdl_tcn, df_x_train, df_y_train



def online_portfolie(action, act_price, 
                     act_portfolio, act_stocks,
                     action_flag, state,
                     price_since_action_in,
                     external_stopp,
                     swing):
    
    '''
    Parameter definition:
        new_stocks = new amount of stocks
        new_portfolio = new portfolio value
        action_flag = shows, if there are exist stocks in your portfolio (False=no stocks)
        state = shows short (=1) or long (=2) position
    
    '''
    
    print("Amount of stocks {}".format(act_stocks))
    
    
    # Check for stopp_strategy and external stopp
    price_since_action, action_stopp_stratgy = stopp_strategy(action, 
                                                              act_price, 
                                                              state, 
                                                              price_since_action_in,
                                                              swing)
    
    if action_stopp_stratgy == 3 or external_stopp:
        
        new_portfolio = (act_price * act_stocks) - order_cost
        new_stocks = 0
        action_flag = False
        state = 0
        
        return new_portfolio, new_stocks, action_flag, state, price_since_action
        
    
    
    
    
    # Buy - empty portfolio -> going long
    if action == 2 and action_flag == False:
        
        new_stocks = (act_portfolio - order_cost)/act_price
        new_portfolio = new_stocks * act_price      
        action_flag = True
        state = 2
        
        
    # Hold or Sell/Buy - no empty portfolio -> going long
    elif action == 2 and action_flag == True:
        
        # sell first and than buy - you were short before
        if state == 1:

            # add profit and sell
            delta = act_portfolio - (act_stocks * act_price)
            new_portfolio = act_portfolio + delta - order_cost
            
            # buy
            new_stocks = (new_portfolio - order_cost)/act_price
            new_portfolio = new_stocks * act_price   
            
                    
        elif state == 2:
            new_stocks = act_stocks
            new_portfolio = new_stocks * act_price
        
        state = 2
        
        
        
    # No action - hold stocks
    elif action == 0 and action_flag == True:
        
        new_portfolio = act_portfolio
        new_stocks = act_stocks
        
    # No action - empty portfolio
    elif action == 0 and action_flag == False:
        
        new_portfolio = act_portfolio
        new_stocks = 0
        
        
        
        
    # Buy - empty portfolio -> going short 
    elif action == 1 and action_flag == False:
        
        new_stocks = (act_portfolio - order_cost)/act_price
        new_portfolio = new_stocks * act_price  
        action_flag = True
        state = 1
        
        
    # Hold/Sell/Buy - no empty portfolio -> going short
    elif action == 1 and action_flag == True:
        
         # sell first and than buy - you were long before
         if state == 2:
             
             # add profit and sell
             delta = (act_portfolio - (act_stocks * act_price)) * (-1)
             new_portfolio = act_portfolio + delta - order_cost
             
             # buy
             new_stocks = (new_portfolio - order_cost)/act_price
             new_portfolio = new_stocks * act_price   
             
         # no action   
         elif state == 1:
             new_stocks = act_stocks
             new_portfolio = new_stocks * act_price
             
         state = 1
         action_flag = True
               
        
    return new_portfolio, new_stocks, action_flag, state, price_since_action
    



def make_forecast(mdl_lstm, mdl_tcn, X):
    return np.argmax(mdl_lstm.predict(X), axis=-1)[0], mdl_tcn.predict(X)[0][0]
    
def update_mdl(model_lstm, model_tcn, X_update, Y_en_update, Y_swing):
    
    updates = 5
    for i in range(updates):
        hist1 = model_lstm.fit(X_update, Y_en_update, epochs=1, batch_size=16, verbose=0, shuffle=False)
        model_tcn.fit(X_update, Y_swing, epochs=1, batch_size=16, verbose=0, shuffle=False)
        
        #model.reset_states()
    
    logging.info('Update model done, train accuracy {}'.format(hist1.history['accuracy'][0]))
    return model_lstm, model_tcn
    




def online_train(lstm_, tcn_, df):
    '''
    Main routine for testing 

    Parameters
    ----------
    mdl : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    act_portfolio : TYPE
        DESCRIPTION.

    '''
    
    act_mdl_lstm = lstm_
    act_mdl_tcn = tcn_
    act_portfolio = money
    act_stocks = 0
    action_flag = False
    state = 0
    psa = 0   
    track_dict = defaultdict(list)


    #ts = pd.Timestamp(dt.datetime.now())
    #ts + BDay(5)
    
    for day in df[end_train_date:end_date].index:
        
        
        # 1. DO FORECASE
        
        # Need to detect trading days (remove weekend and off days)
        shift = 11 # in days
        df_x_forecast, df_y_forecast, df_y_swing = preprocessing(df[day - BDay(shift):day])
        X_forecast, Y_en_forecast, Y_s_forecast = prepare_data(df_x_forecast, df_y_forecast, df_y_swing)
        
        action, swing = make_forecast(act_mdl_lstm, act_mdl_tcn, X_forecast)
        logging.info("Forecast action {} and swing {} for day {}".format(action, round(swing,4), day))
        
        act_stock_value = df[day - pd.DateOffset(days = shift):day]['^GSPC_close'].values[-1]

        logging.info("Stock price {}".format(act_stock_value))
        
        # TRACKING
        track_dict['day'].append(day)
        track_dict['stock_price'].append(act_stock_value)
        track_dict['forecast'].append(df_y_forecast)
        track_dict['swing'].append(round(swing,4))
        track_dict['action_raw'].append(action)
        track_dict['portfolio'].append(act_portfolio)
        track_dict['state'].append(state)
        track_dict['action_flag'].append(action_flag)
        track_dict['psa'].append(psa)

        # Take swings into account -> just use action (long/short) if we expect a swing
        action, swing_action, external_stopp = combine_swing(action, swing, state)
        
        track_dict['action'].append(action)
        track_dict['external_stopp'].append(external_stopp)


        # 2. Update Portfolio
        new_portfolio, new_stocks, new_action_flag, new_state, new_psa = online_portfolie(action, 
                                                                             act_stock_value, 
                                                                             act_portfolio, 
                                                                             act_stocks,
                                                                             action_flag, 
                                                                             state,
                                                                             psa,
                                                                             external_stopp,
                                                                             swing)
        
        track_dict['portfolio_update'].append(new_stocks*act_stock_value)

        
        logging.info("Portfolio status {}, {} , {}, {}".format(new_portfolio, new_stocks, new_action_flag, new_state))
        print("Portfolio status {} , {} , {} , {} , {} , {}".format(round(new_portfolio,3), 
                                                            new_stocks, 
                                                            new_action_flag, 
                                                            new_state, 
                                                            round(swing,3),
                                                            int(new_stocks*act_stock_value)
                                                            )
                                                          )
        

        # 3. Update model with new day
        df_x_train, df_y_train, df_y_swing_train = preprocessing(df[start_date:day])
        X, Y_en, Y_s = prepare_data(df_x_train, df_y_train, df_y_swing_train)
        
        model_re_lstm, model_re_tcn = update_mdl(act_mdl_lstm, act_mdl_tcn, X, Y_en, Y_s)

        #if action[0] == 2:
        #    break
        
        # 4. Update values
        act_mdl_lstm = model_re_lstm
        act_mdl_tcn = model_re_tcn

        act_portfolio = new_portfolio
        act_stocks = new_stocks
        action_flag = new_action_flag
        state = new_state
        psa = new_psa
        
        logging.info("---------------------------------------")
        
    return act_portfolio, track_dict
  
    
  
    

def execute():
    
    
    raw_dict = load_feature_data()
    
    raw_df = raw_dict['^GSPC']
    
    ticker = '^GSPC'
    
    # 1. Init train 
    init_mdl_lstm, init_mdl_tcn, df_x, df_y = init_train( raw_df[start_date:end_train_date], 
                                                         True)
    
    # 2. Calc baseline (= reference performance)
    ref_profit, value_baseline = calc_baseline_profit(raw_df[end_train_date:end_date], 
                                                      ticker)
    
    # 3. Online training/update
    value, tracker_dict = online_train(init_mdl_lstm, 
                                       init_mdl_tcn, 
                                       raw_df[start_date:end_date]
                                       )
    
    return value, tracker_dict
    
    
    
    
if __name__ == '__main__':
    
	value, tracker_dict = execute() 
    
    
    
    