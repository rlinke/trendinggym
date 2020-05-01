# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:35:06 2020

@author: mk
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pickle

from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.optimizers import Adam

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler



# Paramater
path = "./data/stock_feature_data.csv"

init_train_data = pd.Timestamp("2016-01-01")


with open("./data/stock_feature_data.pickle", 'rb') as handle:
    feature_dict = pickle.load(handle)

df_read = feature_dict['^GSPC']#pd.read_csv(path, index_col=0, header=0)


def preprocessing(df):
    
    # not finish
    x_list = ['^GSPC_macd','^GSPC_roc','^GSPC_wr','^GSPC_mov','^GSPC_rsi','^GSPC_close']
    
    df_train = df.dropna()
    
    # Calc percentage of next day based on actual day with > 0.5 %
    df_y_train = df_train['diff']/df_train['0']

    diff = 0.005
    df_y_train = (df_y_train > diff) * 2 + (df_y_train < -diff) * 1
    
    df_x_train = df_train[x_list]
    #df_x_train = df_train[[col for col in df.columns if col not in df_y_train.columns]]
    
    return df_x_train, df_y_train


'''
 Explanation for data preparation
 https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
 
 Input:
     
     F1 = [1, 2, 3, 4, 5]
     F2 = [9, 29, 35, 40, 60]
     
     OUT = [0, 1, 2, 3, 4, 5]
     
     n_steps_in = 3
     X_lstm = [[1, 9], [2, 29], [3, 35]]
              [[2, 29], [3, 35], [4, 40]]
              [[3, 35], [4, 40], [5, 60]]
              
     n_steps_out = 1
     Y_lstm = [1,2,3,4,5]
    
'''

# split a multivariate sequence into samples - for input space
def split_sequences_lstm(sequences, n_steps_in):
    
	X = list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x = sequences[i:end_ix, :]
		X.append(seq_x)
        
	return np.array(X)


def mdl_lstm(n_steps, n_features, dropout1, dropout2):
    
    model = Sequential()
    model.add(LSTM(20, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(20, activation='tanh', return_sequences=True, recurrent_dropout = dropout1))
    model.add(LSTM(10, activation='tanh', recurrent_dropout = dropout2))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
     
    return model

# Data preprocessing
df_X, df_Y = preprocessing(df_read)

n_steps_in = 5 # = lock back
n_features = df_X.shape[1]

# Normalization
scaler = StandardScaler()
df_X_values = scaler.fit_transform(df_X)

# LSTM preprocessing
X = split_sequences_lstm(df_X_values, n_steps_in)

Y = df_Y.values[n_steps_in-1:]
Y = Y.reshape((len(Y), 1))

# One-hot-encoding
Y_en = to_categorical(Y)


# Implement sample weights for training
# from numpy import zeros, newaxis

# unique, counts = np.unique(Y, return_counts=True)
# weights = counts/len(Y)

# weights_list = []
# for el in Y:
#     weights_list.append(weights[el])

# weights_array = np.array(weights_list)


# Create and train model
dropout1 = 0.4
dropout2 = 0.2
model = mdl_lstm(n_steps_in, n_features, dropout1, dropout2)

rlrop = ReduceLROnPlateau(monitor='train_loss', factor=0.1, patience=10, min_delta=0.001)


history = model.fit(X,Y_en, epochs=400,
                    #validation_data=(X_validate,y_validate),
                    shuffle = False,
                    batch_size = 32,
                    #sample_weight = weights_array,
                    callbacks=[rlrop],
                    verbose = 2)


from sklearn.metrics import accuracy_score

class_preds = np.argmax(model.predict(X), axis=-1)

accuracy_score(Y, class_preds)





