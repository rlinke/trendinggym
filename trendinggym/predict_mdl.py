# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:35:06 2020

@author: mk
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.models import Model, Sequential
from keras.layers import LSTM, Dense


# Paramater
path = "./data/stock_feature_data.csv"

init_train_data = pd.Timestamp("2016-01-01")

df_read = pd.read_csv(path, index_col=0, header=0)

def preprocessing(df):
    
    df_train = df.dropna()
    
    df_close_temp = df_train[[s for s in df_train.columns if 'close' in s]]
    
    df_y_train = pd.DataFrame()
    
    df_y_train['+7'] = df_train['+7'].values - df_close_temp.values[:,0]
    df_y_train['+6'] = df_train['+6'].values - df_close_temp.values[:,0]
    df_y_train['+5'] = df_train['+5'].values - df_close_temp.values[:,0]
    df_y_train['+4'] = df_train['+4'].values - df_close_temp.values[:,0]
    df_y_train['+3'] = df_train['+3'].values - df_close_temp.values[:,0]
    df_y_train['+2'] = df_train['+2'].values - df_close_temp.values[:,0]
    df_y_train['+1'] = df_train['+1'].values - df_close_temp.values[:,0]
    
    df_y_train = (df_y_train > 0) * 1
    
    df_x_train = df[[col for col in df.columns if col not in df_y_train.columns]]
    
    return df_x_train, df_y_train,
    

def create_lstm_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


def mdl_lstm(look_back, features):
    
    n_first_layer = 128
    n_second_layer = 64
    n_output_layer = 7
    
    #Build the model
    model = Sequential()
    model.add(LSTM(n_first_layer, input_shape=(look_back,1), return_sequences=True))
    model.add(LSTM(n_second_layer,input_shape=(n_first_layer,1), dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(n_output_layer, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

df_X, df_Y = preprocessing(df_read)

look_back = 1
trainX, trainY = create_lstm_dataset(train, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

history = model.fit(X_train,y_train,
                    epochs=200,
                    validation_data=(X_validate,y_validate),
                    shuffle=True,
                    batch_size=2, 
                    verbose=2)


