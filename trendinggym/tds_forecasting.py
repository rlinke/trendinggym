# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:33:14 2020

@author: Richard

recreating the stock forecasting tds article
# https://towardsdatascience.com/using-deep-learning-ai-to-predict-the-stock-market-9399cf15a312

links on stateful lstm:
    https://stackoverflow.com/questions/53190253/stateful-lstm-and-stream-predictions
    https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html
"""


# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
plt.style.use("bmh")

# Technical Analysis library
import ta

# Neural Network library
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from trendinggym.tds_helper import * 

from trendinggym.modelpool import build_lstm_model, build_tcn_model


"""
    LOAD AND PREPROCESS

"""
## Creating the NN
def load_dataset(filepath):
    # Loading in the Data
    df = pd.read_csv(filepath, index_col=0)
    
    ## Datetime conversion
    df.index = pd.to_datetime(df.index.values)
    
    # Setting the index
    # df.set_index('Date', inplace=True)
    
    # Dropping any NaNs
    df.dropna(inplace=True)
    
    
    
    ## Technical Indicators
    
    # Adding all the indicators
    df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    
    # Dropping everything else besides 'Close' and the Indicators
    df.drop(['open', 'high', 'low', 'adjclose', 'volume', 'ticker'], axis=1, inplace=True)
    
    return df

def setup_dataset(is_cat, df, y_in, input_shape, output_shape):      
    ## Scaling
    # Scale fitting the close prices separately for inverse_transformations purposes later
    close_scaler = RobustScaler()
    
    close_scaler.fit(df[['close']])
    
    # Normalizing/Scaling the DF
    scaler = RobustScaler()
    
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    # input_shape, output_shape = 30, 1
    # Splitting the data into appropriate sequences
    seq = df.values
    # Creating a list for both variables
    X, y = [], []
    i=0
    for i in range(len(seq)):
        
        # Finding the end of the current sequence
        end = i + input_shape
        out_end = end + output_shape
        
        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(seq):
            break
        
        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        if is_cat:
            seq_x, seq_y = seq[i:end, :], y_in[end,:]
        else:
            seq_x, seq_y = seq[i:end, :], seq[end:out_end]
        
        X.append(seq_x)
        y.append(seq_y)
    
    x,y = np.array(X), np.array(y)        
    #split in train and validation data
    # X_train, X_val,y_train, y_val = train_test_split(X,y, shuffle=False)
    # permutate the train data 
    
    """
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    """
    return df, x, y, close_scaler


"""
    TRAIN THE MODEL!
"""


def train_init(model, init_interval, x, y, cbs=None):
    #baseline training with x%
    x_t = x[0:init_interval,:,:]
    y_t = y[0:init_interval]
        
    hist = model.fit(x_t, 
                     y_t, 
                    epochs=150, 
                    shuffle=False,
                    batch_size=64,
                    # validation_split=0.1,
                    callbacks=cbs
                )

    return hist, model


"""
iteration = 0
lookback_interval = 5
features = options["features"]"""
def predict_one_step(model, x, iteration, lookback_interval, features):
    #model, df, iteration, lookback_interval, features
    #iteration, lookback_interval, features = i, options["lookback_interval"], options["features"]
    if isinstance(x, pd.DataFrame):
        x_ = x.iloc[iteration]
    else:
        x_ = x[iteration,:,:]

    # Predicting using rolling intervals
    yhat = model.predict(x_.reshape(1, lookback_interval, features))

    # Transforming values back to their normal prices
    yhat = close_scaler.inverse_transform(yhat)[0]
    ytrue = close_scaler.inverse_transform(np.array(x[iteration+1,-1,0]).reshape(1,-1))[0][0]

    # DF to store the values and append later, frequency uses business days
    return (df.index[iteration+lookback_interval], yhat[-1], ytrue)
    

def predict_categorical(model, x, iteration, lookback_interval, features):
    #model, df, iteration, lookback_interval, features
    #iteration, lookback_interval, features = i, options["lookback_interval"], options["features"]
    if isinstance(x, pd.DataFrame):
        x_ = x.iloc[iteration]
    else:
        x_ = x[iteration,:,:]

    # Predicting using rolling intervals
    yhat = model.predict(x_.reshape(1, lookback_interval, features))
    yhat = np.argmax(yhat)
    return (df.index[iteration+lookback_interval], yhat) 


def retrain(model, i, replay_buffer, x, y):
    if replay_buffer == 0:
        start = 0
    else:
        start = i-replay_buffer if i-replay_buffer > 0 else 0
    x_t = x[start:i,:,:]
    y_t = y[start:i]
    history = model.fit(x_t,
                    y_t, 
                    epochs=3, 
                    batch_size=64, 
                    shuffle=False,
                    verbose=0,
                    validation_split=0.1)
    return history, model


# train, predict for each step after the init interval, 
# do not go further than  we can becase we have no further data
def loop(model,start, end, x, df, options, train=True, verbose=2, result_cb=None):
    results = []
    last_possible_timestep = len(x)
    
    if end == -1:
        end = last_possible_timestep
    
       
    end = min(end, last_possible_timestep)
    
    start = max(start, options["lookback_interval"])
    
    if verbose>=1:
        print("-"*40)
        if train:
            print("TRAINING AND EVALUATION")
        else:
            print("ONLY EVALUATION")
        print(f"iterating over timespan {df.index[start]}[{start}] to {df.index[end]}[{end}]")
        print("")
    
    for i in range(start, end):
        # print the current timetamp
        # ts, pred, ytrue = predict_one_step(model, x, i, options["lookback_interval"], options["features"])
        ts, pred = predict_categorical(model, x, i, options["lookback_interval"], options["features"])
        
        actual_close = df["close"].iloc[i+options["lookback_interval"]]
        result = [ts, pred, actual_close]
        results.append(result)
        
        if result_cb:
            result_cb(*result)
            
        if verbose >= 2:
            print(f"predicting: {df.index[i]}:\t{pred:.4f} | {actual_close:.4f}")
        
        if train:
            _, model = retrain(model, i, replay_buffer=0, x=x, y=y)

    return results, model

"""
    PLOT RESULTS
"""

def plot_result(results):
    s_ts = results[0][0]
    e_ts = results[-1][0]
    
    tss = [r[0] for r in results]
    prs = [r[1] for r in results]
    ars = [r[2] for r in results]
    
    results = pd.DataFrame(np.array([prs, ars]).T, 
                          index=tss, 
                          columns=["prediction", "actual"])
    # Plotting
    
    plt.plot(results["actual"], label='Actual')
    plt.plot(results["prediction"], 'r.', label='Predicted')
    
    if init_interval < len(results):
        plt.axvspan(results.index.values[init_interval],
                    results.index.values[-1],
                    alpha=0.2
                    )
    
    plt.title(f"Predicted vs Actual Closing Prices")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    


#%%


options = {
    "filepath": "data/stocks/^GSPC.csv",
    "lookback_interval":20,
    "features" : "auto",
    "is_categorical": True,
    "out": 3
}

training_fraction = 0.4

df = load_dataset(options["filepath"])

df_orig = df.copy()


trunc_features = True
if trunc_features:
    # macd roc wr mov rsi close
    df = df.filter(["close", "trend_macd","momentum_roc",  "momentum_wr", "volume_em", "momentum_rsi"], axis=1)

is_category = True
if is_category:
    
    temp = (df_orig["close"] - df_orig["close"].shift(1)).shift(-1)/df_orig["close"]
    
    
    threshold = 0.005
    mask_pos = temp > threshold
    
    mask_neg = temp < -threshold
    
    # set all to 0 - no action
    temp[:] = 0
    # set all positive to 2 - long option
    temp[mask_pos] = 2
    # set all negative to 1 - short option
    temp[mask_neg] = 1
    temp = to_categorical(temp)
    df, x, y, close_scaler = setup_dataset(True, df, temp, options["lookback_interval"], options["out"])

else:
    df, x, y, close_scaler = setup_dataset(False, df, None, options["lookback_interval"], options["out"])




init_interval = int(len(x) * training_fraction)
# set features correctly
if options["features"] == "auto":
    options["features"] = x.shape[-1]


# eof2016 = pd.Timestamp("2016-12-30")
# eof2016_index = df.index.get_loc(eof2016)

eof2017 = pd.Timestamp("2017-12-29")
eof2017_index = df.index.get_loc(eof2017)

eof2018 = pd.Timestamp("2018-12-31")
eof2018_index = df.index.get_loc(eof2018)

eof2019 = pd.Timestamp("2019-12-31")
eof2019_index = df.index.get_loc(eof2019)

eofds_index = len(df)

# model = build_lstm_model(**options)
model = build_tcn_model(**options)

rlrop = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_delta=0.00001)
es_cb = EarlyStopping(monitor='acc', min_delta=1e-4, patience=5, verbose=0, restore_best_weights=True)

hist, model = train_init(model, eof2017_index, x, y, cbs=[rlrop, es_cb])

visualize_training_results(hist)

#%%

# run a full simulation run from the start:
# 40 -> because we currently use 40 timesteps. this will be checked in the function anyways
#   here just added for clarity
"""
try:
    results
except NameError:
   results2017, model = loop(model, eof2016_index, eof2017_index,  x, df_orig, options, train=False, verbose=1)
   results2018, model = loop(model, eof2017_index, eof2018_index,  x, df_orig, options, train=False, verbose=1)
   results2019, model = loop(model, eof2018_index, eof2019_index,  x, df_orig, options, train=False, verbose=1)

plot_result(results2017)
plot_result(results2018)
plot_result(results2019) 
"""


# results2017, model = loop(model, eof2016_index, eof2017_index, x, df_orig, options, train=True)
results2018, model = loop(model, eof2017_index, eof2018_index, x, df_orig, options, train=True)
results2019, model = loop(model, eof2018_index, eof2019_index, x, df_orig, options, train=True)
# results2020, model = loop(model, eof2019_index, len(x), x, df_orig, options, train=True)

# plot_result(results2017)
plt.figure()
plot_result(results2018)
plot_result(results2019)
# plot_result(results2020)

baseline_results= pd.concat(pd.DataFrame(f) for f in [results2018, results2019])
baseline_results.columns = ['time', 'predicted', 'actual']
baseline_results.index = baseline_results["time"]
baseline_results.drop("time", inplace=True, axis=1)

baseline_results.to_pickle("data/cache/2020_05_13_baseline.pkl")
#%%


_, _ = loop(model, eof2017_index, eof2018_index, x, df_orig, options, train=False)

# first job: order into classes instead of regression
# second job: create useful prediction / value function for comparability
# fourth job: add additional features (currently only one instrument)
# fifth job: schaue auf anleihen die reagierne manchmal früher - generell siehe daniellos idee: schauen welche märkte nacheilen oder voreilen und dann auf den nacheilenden märkten traden















