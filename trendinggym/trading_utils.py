# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:57:27 2020

@author: mk
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


def load_feature_data():
            
    with open("./data/stock_feature_data.pickle", 'rb') as handle:
        feature_dict = pickle.load(handle)
    
    #df = feature_dict['^GSPC']#pd.read_csv(path, index_col=0, header=0)
    return feature_dict


def money_managment():
    '''
    tbd

    Returns
    -------
    None.

    '''
    
    return None




def combine_swing(action, swing, state):
    
    # Init
    external_stopp = False
    swing_action = 0

    # Set the threshold for the swing predictor
    s_thr = 0.03
    
    # If we own stocks, sell in case the swing predictor is indicating a high contra signal
    
    if state == 1 and swing > (s_thr*2): # if we are short
        external_stopp = True
    elif state == 2 and swing < (s_thr*2): # if we are long
        external_stopp = True
    
    
    # Prove a buy action
    if action == 2:
        if swing < s_thr:
            action = 0 #overwrite
        else:
            swing_action = 2 # accept action
            
    # Prove a short action
    if action == 1:
        if swing > -s_thr:
            action = 0 #overwrite
        else:
            swing_action = 1 # accept action
            
    return action, swing_action, external_stopp




def stopp_strategy(action, act_price, state, price_since_action, swing):
    '''
    Check if your loss is below threshold (loss protection) -> sell
    Check if the swing prediction indicate a change and we are positiv -> sell

    Parameters
    ----------
    action : TYPE
        DESCRIPTION.
    act_price : TYPE
        DESCRIPTION.
    state : TYPE
        DESCRIPTION.
    price_since_action : TYPE
        DESCRIPTION.

    Returns
    -------
    price_since_action : TYPE
        DESCRIPTION.
    action_stopp_stratgy : TYPE
        DESCRIPTION.

    '''
    
    threshold_p = 0.10
    threshold_n = 0.10
    swing_thr = 0.3
    profit = 1
    action_stopp_stratgy = 0

    # save last price
    if action != 0:
        price_since_action = act_price
        
    else:
        # Long position -> calc profit
        if state == 2:
            profit = (act_price/price_since_action) 
            
        # Short position -> calc profit
        elif state == 1: 
            profit = (price_since_action/act_price)
        
        # If profit is below threshold -> sell
        if profit < (1 - threshold_n):
            # Need to sell stocks
            logging.info('---> Need to sell stocks due to stopp stratagy')
            action_stopp_stratgy = 3
            
        # Check swing indicator and actual profit to sell gains
        if profit > (1 + threshold_p):
            
            # Long position
            if state == 2:
                if swing < -swing_thr:
                    logging.info('---> Need to sell stocks due to stopp stratagy - long')
                    action_stopp_stratgy = 3
                    
            # short position
            elif state == 1:
                if swing > -swing_thr:
                    logging.info('---> Need to sell stocks due to stopp stratagy - short')
                    action_stopp_stratgy = 3
                
    
    return price_since_action, action_stopp_stratgy


def plot_tracking(dict_):
    
    
    
    sp = tracker_dict['stock_price']
    t = tracker_dict['day']
    
    plt.plot(sp,t)
