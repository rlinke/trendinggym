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


def load_feature_data():
            
    with open("./data/stock_feature_data.pickle", 'rb') as handle:
        feature_dict = pickle.load(handle)
    
    df = feature_dict['^GSPC']#pd.read_csv(path, index_col=0, header=0)

    return df 

def calc_swings(df):
    
    
    return None


def preprocessing(df):
    
    # not finish
    x_list = ['^GSPC_macd','^GSPC_roc','^GSPC_wr','^GSPC_mov','^GSPC_rsi','^GSPC_close']
    
    df_train = df.dropna()
    
    # Calc percentage of next day based on actual day with > 1 %
    df_y_train = df_train['diff']/df_train['0']

    diff = 0.01
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
    model.add(LSTM(50, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='tanh', recurrent_dropout = dropout1))
    #model.add(LSTM(10, activation='tanh', recurrent_dropout = dropout2))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])
     
    return model


def prepare_data(df_X, df_Y):
    
    # Data preprocessing
    #df_X, df_Y = preprocessing(df)
    
    n_steps_in = 5 # = lock back
    n_features = 6#df_X.shape[2]
    
    # Normalization
    scaler = StandardScaler()
    df_X_values = scaler.fit_transform(df_X)
    
    # LSTM preprocessing
    X = split_sequences_lstm(df_X_values, n_steps_in)
    
    Y = df_Y.values[n_steps_in-1:]
    Y = Y.reshape((len(Y), 1))
    
    # One-hot-encoding
    Y_en = to_categorical(Y)
    
    return X, Y_en


# Implement sample weights for training
# from numpy import zeros, newaxis

# unique, counts = np.unique(Y, return_counts=True)
# weights = counts/len(Y)

# weights_list = []
# for el in Y:
#     weights_list.append(weights[el])

# weights_array = np.array(weights_list)


def create_train_mdl(X, Y_en):
    
    
    n_steps_in = 5 # = lock back
    n_features = 6#X.shape[2] # Note: 2 -> lstm format
    
    # Create and train model
    dropout1 = 0.4
    dropout2 = 0.2
    model = mdl_lstm(n_steps_in, n_features, dropout1, dropout2)
    
    rlrop = ReduceLROnPlateau(monitor='train_loss', factor=0.1, patience=10, min_delta=0.01)
    
    
    hist = model.fit(X,Y_en, 
                        epochs=400,
                        #validation_data=(X_validate,y_validate),
                        shuffle = False,
                        batch_size = 16,
                        #sample_weight = weights_array,
                        callbacks=[rlrop],
                        verbose = 2)
    
    #to do save mdl
    
    return model 



def make_forecast_test(model, X):
    return np.argmax(model.predict(X), axis=-1), model.predict(X)

def plotting(mdl, stock_dict, feature_dict, ticker):
    
    
    import finplot as fplt
    
    filepath = "./data/stock_chart_data.pickle"
    save_path = "./data/stock_feature_data.pickle"
            
    ticker = "^GSPC"


    with open(filepath, 'rb') as handle:
        stock_dict = pickle.load(handle)

    with open(save_path, 'rb') as handle:
        feature_dict = pickle.load(handle)
    
    mdl = init_mdl
    
    
    
         
    df_x, df_y = preprocessing(feature_dict[ticker][start_date:end_date])
    X_forecast, Y_en_forecast = prepare_data( df_x, df_y)
    
    action, prob = make_forecast_test(mdl, X_forecast)


    df = stock_dict[ticker][df_x.index[0]:df_x.index[-1]].copy()
    
    df['time'] = df.index
    df = df.astype({'time':'datetime64[ns]'})
    df = df.reset_index()


    df_f = feature_dict[ticker][df_x.index[0]:df_x.index[-1]].dropna()
    
    
    
    
    
    # create three plots
    ax,ax2,ax3 = fplt.create_plot(ticker, rows=3)
    
    # plot candle sticks
    #candles = df[['time','Open_^GSPC','Close_^GSPC','High_^GSPC','Low_^GSPC']]
    #fplt.candlestick_ochl(candles, ax=ax)
    
    
    fplt.plot(df_x.index, df['Close_^GSPC'], ax=ax, color='#927', legend='action')

    
    fplt.plot(df_x.index, action, ax=ax2, color='#927', legend='action')
    fplt.plot(df_x.index, np.max(prob, axis=-1), ax=ax3, color='#927', legend='prob')
    
    
    # create three plots
    ax,ax2,ax3,ax4,ax5,ax6 = fplt.create_plot(ticker, rows=6)
    
    # plot candle sticks
    #candles = df[['time','Open_^GSPC','Close_^GSPC','High_^GSPC','Low_^GSPC']]
    #fplt.candlestick_ochl(candles, ax=ax)
    
    col = ['^GSPC_macd', '^GSPC_roc', '^GSPC_wr', '^GSPC_mov', '^GSPC_rsi']
    fplt.plot(df_f.index, df_f['^GSPC_macd'].values, ax=ax, color='#927', legend='macd')
    fplt.plot(df_f.index, df_f['^GSPC_roc'].values, ax=ax2, color='#927', legend='roc')
    fplt.plot(df_f.index, df_f['^GSPC_wr'].values, ax=ax3, color='#927', legend='wr')
    fplt.plot(df_f.index, df_f['^GSPC_mov'].values, ax=ax4, color='#927', legend='mov')
    fplt.plot(df_f.index, df_f['^GSPC_rsi'].values, ax=ax5, color='#927', legend='rsi')

    fplt.plot(df_f.index, df_f['diff'].values, ax=ax6, legend='diff')


    # create three plots
    ax,ax2 = fplt.create_plot(ticker, rows=2)
    
    # plot candle sticks
    #candles = df[['time','Open_^GSPC','Close_^GSPC','High_^GSPC','Low_^GSPC']]
    #fplt.candlestick_ochl(candles, ax=ax)
    
    fplt.plot(df_x.index, df['Close_^GSPC'], ax=ax, color='#927', legend='action')

    
    fplt.plot(df_f.index, df_f['diff'].values, ax=ax2, legend='diff')

    
# from sklearn.metrics import accuracy_score

# class_preds = np.argmax(model.predict(X), axis=-1)

# accuracy_score(Y, class_preds)



