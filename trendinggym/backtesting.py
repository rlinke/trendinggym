# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:36:19 2020

@author: mk
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from predict_mdl import load_feature_data, preprocessing, prepare_data, create_train_mdl




''' Define global Timeline and Parameters'''

start_date = pd.Timestamp("2010-01-01")

end_train_date = pd.Timestamp("2018-01-01")

end_date = pd.Timestamp("2019-01-01")

order_cost =  5

money = 10000

''' ---------------------------------------------------------  '''



def calc_baseline_profit(df, ticker):
    '''
    Calculate basline balance  and profit

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
    

def init_train(df):
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
    
    df_x_train, df_y_train = preprocessing(df)
    X, Y_en = prepare_data(df_x_train, df_y_train)
    init_mdl = create_train_mdl(X, Y_en)
    
    print('Trained init model')
    
    return init_mdl, df_x_train, df_y_train


def money_managment():
    '''
    Check if loss is getting to high and than sell

    Returns
    -------
    None.

    '''
    
    return None

def stopp_strategy(action, act_price, state, price_since_action):
    
    
    threshold = 0.1
    profit = 1
    action_stopp_stratgy = 0

    
    if action != 0:
        price_since_action = act_price
        
    else:
        # Long position
        if state == 2:
            profit = (act_price/price_since_action) + threshold
            
        # Short position
        elif state == 1: 
            profit = (price_since_action/act_price) + threshold
        
        
        if profit < 1:
            # Need to sell stocks
            print('Need to sell stocks due to stopp stratagy')
            action_stopp_stratgy = 3
    
    return price_since_action, action_stopp_stratgy


def online_portfolie(action, act_price, 
                     act_portfolio, act_stocks,
                     action_flag, state):
    
    '''
    Parameter definition:
        new_stocks = new amount of stocks
        new_portfolio = new portfolio value
        action_flag = shows, if there are exist stocks in your portfolio (False=no stocks)
        state = shows short (=1) or long (=2) position
    
    '''
    
    print("Amount of stocks ", act_stocks)
    
    
    # # Check for stopp_strategy
    # price_since_action, action_stopp_stratgy = stopp_strategy(action, 
    #                                                           act_price, 
    #                                                           state, 
    #                                                           price_since_action)
    
    # if action_stopp_stratgy == 3:
    #     return new_portfolio, new_stocks, action_flag, state
        
    
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
        
        
        
    # Sell - depending on short or long position
    elif action == 0 and action_flag == True:
        
        new_portfolio = act_portfolio
        new_stocks = act_stocks
        
        # # short
        # if state == 1:
        #     # add 
        #     delta = (act_portfolio - (act_stocks * act_price)) #* (-1)
        #     new_portfolio = act_portfolio + delta
        # # long 
        # elif state == 2:
        #     # add profit
        #     delta = (act_portfolio - (act_stocks * act_price)) * (-1)
        #     new_portfolio = act_portfolio + delta
            
            
        #new_stocks = 0
        #action_flag = False
            
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
               
        
    return new_portfolio, new_stocks, action_flag, state
    

def make_forecast(model, X):
    return np.argmax(model.predict(X), axis=-1)
    


def update_mdl(model, X_update, Y_en_update):
    
    updates = 5
    for i in range(updates):
        hist = model.fit(X_update, Y_en_update, epochs=1, batch_size=16, verbose=0, shuffle=False)
        #model.reset_states()
    
    print('Update model done, train accuracy ', hist.history['accuracy'])
    return model
    





def online_train(mdl, df):
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
    
    act_mdl = mdl 
    act_portfolio = money
    act_stocks = 0
    action_flag = False
    state = 0
       
    for day in df[end_train_date:end_date].index:
        
        
        # 1. DO FORECASE
        
        # Need to detect trading days (remove weekend and off days)
        shift = 7
        ddays = len(df[day - pd.DateOffset(days = shift):day])
        if ddays == 6:
            shift = 6
        elif ddays == 4:
            shift = 8
        
        df_x_forecast, df_y_forecast = preprocessing(df[day - pd.DateOffset(days = shift):day])
        X_forecast, Y_en_forecast = prepare_data(df_x_forecast, df_y_forecast)
        
        action = make_forecast(act_mdl, X_forecast)
        print("Forecast action {} for day {}".format(action[0], day))
        
        act_stock_value = df[day - pd.DateOffset(days = shift):day]['^GSPC_close'].values[-1]

        print("Stock price ", act_stock_value)        
        
        # 2. Update Portfolio
        new_portfolio, new_stocks, new_action_flag, new_state = online_portfolie(action[0], 
                                                                             act_stock_value, 
                                                                             act_portfolio, 
                                                                             act_stocks,
                                                                             action_flag, 
                                                                             state)

        
        print("Portfolio status {}, {} , {}, {}".format(new_portfolio, new_stocks, new_action_flag, new_state))
        # 3. Update model with new day
        df_x_train, df_y_train = preprocessing(df[start_date:day])
        X, Y_en = prepare_data(df_x_train, df_y_train)
        
        model_return = update_mdl(act_mdl, X, Y_en)

        #if action[0] == 2:
        #    break
        
        # 4. Update values
        act_mdl = model_return
        
        act_portfolio = new_portfolio
        act_stocks = new_stocks
        action_flag = new_action_flag
        state = new_state
        
        print("---------------------------------------")
        
    return act_portfolio
        #print(day)
  
    
  
    

def execute():
    
    
    raw_df = load_feature_data()
    
    ticker = '^GSPC'
    
    # 1. Init train 
    init_mdl, df_x, df_y = init_train( raw_df[start_date:end_train_date])
    
    # 2. Calc baseline (= reference performance)
    ref_profit, value_baseline = calc_baseline_profit(raw_df[end_train_date:end_date], ticker)
    
    # 3. Online training/update
    value = online_train(init_mdl, raw_df[start_date:end_date])
    
    return value
    
    
    
    
if __name__ == '__main__':
    
	value = execute() 
    
    
    
    