# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:35:06 2020

@author: mk
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pickle

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler

# For LSTM
from keras.callbacks import ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Conv1D, Dropout, Lambda, concatenate
from keras.utils import to_categorical
from keras.optimizers import Adam


# For TCN
from tensorflow.keras.layers import Dense as tfDense
from tensorflow.keras import Input, Model
from tensorflow.keras import Model as tfModel

from tcn import TCN, tcn_full_summary



# Paramater
path = "./data/stock_feature_data.csv"

init_train_data = pd.Timestamp("2016-01-01")



def calc_swings(df):
    '''
    calc metric for swing

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    swing_cols = ['+7', '+6', '+5', '+4', '+3', '+2']
    
    #Calc diff
    df_diff = pd.DataFrame()
    
    for col in swing_cols:
        df_diff[col] = 1 - df[col]/df['0']
        
    df_calc = pd.DataFrame(columns=['swing'])#, index=df_diff.index)
    
    for idx, val in df_diff.iterrows():
        df_calc.loc[idx] = [np.sum(val)]
        
    return df_calc


def preprocessing(df):
    
    # Features -> better solution 
    x_list = ['^GSPC_macd','^GSPC_roc','^GSPC_wr','^GSPC_mov','^GSPC_rsi',
              '^GSPC_close','^GSPC_open','^GSPC_high','^GSPC_low','^GSPC_vol']
    
    df_train = df.dropna()
    
    # Calc percentage of next day based on actual day with > 1 %
    df_y_train = df_train['diff']/df_train['0']

    diff = 0.005
    df_y_train = (df_y_train > diff) * 2 + (df_y_train < -diff) * 1
    
    # Get swing indicator
    df_swing =  calc_swings(df_train)
    
    df_x_train = df_train[x_list]
    #df_x_train = df_train[[col for col in df.columns if col not in df_y_train.columns]]
    
    
    return df_x_train, df_y_train, df_swing #pd.concat([df_y_train, df_swing], axis=1)


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


def mdl_lstm(n_steps, batch_size, n_features, dropout1, dropout2):
    
    model = Sequential()
    model.add(LSTM(50, activation='tanh', 
                   stateful=False,
                   return_sequences=True, 
                   input_shape=(n_steps, n_features)
                   #batch_input_shape=(batch_size, n_steps, n_features)
                   )
              )
    
    model.add(LSTM(50, activation='tanh', 
                   stateful=False, 
                   recurrent_dropout = dropout1)
              )
    
    #model.add(SeqSelfAttention(attention_activation='tanh'))
    model.add(Dense(3, activation='softmax'))     
    
    model.compile(optimizer=Adam(lr=0.01), metrics=['accuracy'], loss='categorical_crossentropy')
                  
    return model



def mdl_tcn(timesteps, input_dim):
    
    #batch_size, timesteps, input_dim = None, timesteps, input_dim
    
    i = Input(batch_shape=(None, timesteps, input_dim))
    
    o = TCN(nb_filters=64, kernel_size=6, 
              return_sequences=True, activation='tanh',
              dropout_rate=0.1)(i)  
    
    o = TCN(nb_filters=64, kernel_size=4, nb_stacks=1, 
              dilations=[1, 2, 4, 8, 16, 32], padding='causal', 
              use_skip_connections=True, dropout_rate=0.4,
              activation='tanh',return_sequences=False)(o)
    
    out = tfDense(1)(o)
    
    model = tfModel(inputs=[i], outputs=[out])
    model.compile(optimizer='adam', loss='mse')
    
    tcn_full_summary(model, expand_residual_blocks=False)
           
    return model


# add these layers to the graph model and connect them to the sequence input
#model.add_node(shared_model, name="shared_layers", input="sequence_input")

# now add your output layers
#model.add_node(Dense(10, activation="softmax"), name="output1", input="shared_layers", create_output=True)
#model.add_node(Dense(2, activation="linear"), name="output2", input="shared_layers", create_output=True)
# ...

# compile the model, potentially with different loss functions per output
#model.compile("rmsprop", {"output1": "categorical_crossentropy", "output2": "mse"})



def prepare_data(df_X, df_Y, df_swing):
    
    # Data preprocessing
    #df_X, df_Y, df_swing = preprocessing(df)
    
    n_steps_in = 10 # = lock back
    n_features = 10 #df_X.shape[2]
    
    # Normalization
    scaler = StandardScaler()
    df_X_values = scaler.fit_transform(df_X)
    
    # LSTM preprocessing
    X = split_sequences_lstm(df_X_values, n_steps_in)
    
    Y = df_Y.values[n_steps_in-1:]
    Y = Y.reshape((len(Y), 1))
    
    # One-hot-encoding
    Y_en = to_categorical(Y)
    
    #Y_f = np.concatenate((Y_en, ), axis=1)
    
    return X, Y_en, df_swing.values[n_steps_in-1:]



# Implement sample weights for training
# from numpy import zeros, newaxis

# unique, counts = np.unique(Y, return_counts=True)
# weights = counts/len(Y)

# weights_list = []
# for el in Y:
#     weights_list.append(weights[el])

# weights_array = np.array(weights_list)


def create_train_mdl(X, Y_en, Y_s):
    
    
    n_steps_in = X.shape[1] # = lock back
    n_features = X.shape[2] # = feature X.shape[2] # Note: 2 -> lstm format
    
    # Create and train model
    dropout1 = 0.4
    dropout2 = 0.2
    bs = 16
    
    # Create LSTM Model
    model_lstm = mdl_lstm(n_steps_in, bs, n_features, dropout1, dropout2)
    
    rlrop = ReduceLROnPlateau(monitor='train_loss', factor=0.1, patience=20, min_delta=0.005)
    
    hist1 = model_lstm.fit(X,Y_en, 
                        epochs=400,
                        shuffle = False,
                        batch_size = bs,
                        #sample_weight = weights_array,
                        callbacks=[rlrop],
                        verbose = 2)
    
    
    # Create TNC Model
    model_tcn = mdl_tcn(n_steps_in, n_features)
    
    hist2 = model_tcn.fit(X, Y_s, 
                          epochs=400, 
                          validation_split=0.0)
    
    
    return model_lstm, model_tcn



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



