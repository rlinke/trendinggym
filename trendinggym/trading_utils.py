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
    
    if state:
        s_thr = 0
    else:
        s_thr = 0.05
    
    
    swing_action = 0
    if action == 2:
        if swing < s_thr:
            action = 0
        else:
            swing_action = 2
    if action == 1:
        if swing > -s_thr:
            action = 0
        else:
            swing_action = 1
            
    return action, swing_action




def stopp_strategy(action, act_price, state, price_since_action):
    '''
    

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
    
    threshold = 0.05
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
            logging.info('---> Need to sell stocks due to stopp stratagy')
            action_stopp_stratgy = 3
    
    return price_since_action, action_stopp_stratgy



