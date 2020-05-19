# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:11:55 2020

@author: Richard
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input

from tcn import TCN # , tcn_full_summary

# lookback_interval, features, out, is_categorical = 20, 40, 3, True

def build_tcn_model(lookback_interval, features, out, is_categorical, **kwargs):
    # Instatiating the model
    
    # Input layer
    inp = Input(batch_shape=(None, lookback_interval, features))
    
    ti = TCN(64, nb_stacks=1, dilations=[1,2,4,8])(inp)
        
    # Output layer
    out = Dense(out, activation='softmax')(ti)

    model = Model(inputs=[inp], outputs=[out])

    
    if is_categorical:    
        model.compile(optimizer="adam", loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else: 
        # Compiling the data with selected specifications
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model



def build_lstm_model(lookback_interval, features, out, is_categorical, **kwargs):

    # Instatiating the model
    model = Sequential()
    
    # Input layer
    model.add(LSTM(50, 
                   activation="tanh", 
                   return_sequences=True, 
                   input_shape=(lookback_interval, features)))
    
    # model.add(LSTM(30, activation="tanh", return_sequences=True, recurrent_dropout = 0.3))

    # Final Hidden layer
    model.add(LSTM(50, activation="tanh", recurrent_dropout = 0.4))
        
    # Output layer
    model.add(Dense(out, activation='softmax'))
    
    if is_categorical:    
        model.compile(optimizer="adam", loss='categorical_crossentropy',
                      metrics=['accuracy'])
    else: 
        # Compiling the data with selected specifications
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model
