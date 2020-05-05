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

from trendinggym.tds_helper import * 


def test_future_leakage(df):
    
    window1 = np.random.randint(0,len(df))
    window2 = window1 + 10 
    
    df1 = df.copy().head(window1)
    df2 = df.copy().head(window2)
    
    df1 = ta.add_all_ta_features(df1, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    df2 = ta.add_all_ta_features(df2, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    df2 = df2.head(window1)
    for col in df1.columns:
        if not (df1[col] == df2[col]).all():
            print(col)
            
    df1["trend_vortex_ind_pos"].plot()
    df2["trend_vortex_ind_pos"].plot()
    
    col="others_dr"
    sum((df2[col]-df1[col])!=0)
    """
    # non conformant features...
    others_dr
    volume_vpt
    trend_vortex_ind_pos
    trend_vortex_ind_neg
    trend_vortex_ind_diff
    trend_trix
    trend_dpo
    trend_kst
    trend_kst_sig
    trend_kst_diff
    trend_visual_ichimoku_a
    trend_visual_ichimoku_b
    """

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


def setup_dataset(df, input_shape, output_shape):      
    ## Scaling
    
    # Scale fitting the close prices separately for inverse_transformations purposes later
    close_scaler = RobustScaler()
    
    close_scaler.fit(df[['close']])
    
    # Normalizing/Scaling the DF
    scaler = RobustScaler()
    
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    # input_shape, output_shape = 30, 1
    # Splitting the data into appropriate sequences
    x, y = split_sequence(df.to_numpy(), input_shape, output_shape)
    
        
    #split in train and validation data
    # X_train, X_val,y_train, y_val = train_test_split(X,y, shuffle=False)
    # permutate the train data 
    
    """
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    """
    return x, y, close_scaler


"""
    TRAIN THE MODEL!
"""

def setup_model(lookback_interval, lookahead_interval, features, out, **kwargs):
    # How many periods looking back to learn
    # How many periods to predict
    # Features 

    # Instatiating the model
    model = Sequential()
    
    # Activation
    activ = "tanh"
    
    # Input layer
    model.add(LSTM(90, 
                   activation=activ, 
                   return_sequences=True, 
                   input_shape=(lookback_interval, features)))
    
    # Hidden layers
    layer_maker(model,
                n_layers=2, 
                n_nodes=30, 
                activation=activ)
    
    # Final Hidden layer
    model.add(LSTM(60, activation=activ))
    
    # Output layer
    model.add(Dense(out))
    
    # Compiling the data with selected specifications
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def train_init(model, init_interval, x, y, cbs=None):
    #baseline training with x%
    x_t = x[0:init_interval,:,:]
    y_t = y[0:init_interval]
    
    hist = model.fit(x_t, 
                     y_t, 
                    epochs=25, 
                    batch_size=64, 
                    validation_split=0.1,
                    callbacks=cbs)

    return hist, model


def predict_one_step(model, x, iteration, lookback_interval, features):
    #model, df, iteration, lookback_interval, features
    #iteration, lookback_interval, features = i, options["lookback_interval"], options["features"]
    if isinstance(x, pd.DataFrame):
        x_ = x.iloc[iteration-lookback_interval:iteration]
    else:
        x_ = x[iteration-lookback_interval:iteration,:,:]

    # Predicting using rolling intervals
    yhat = model.predict(x_)

    # Transforming values back to their normal prices
    yhat = close_scaler.inverse_transform(yhat)[0]

    # DF to store the values and append later, frequency uses business days
    return (df.index[iteration], yhat[-1])
    


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
                    verbose=0,
                    validation_split=0.1,
                    callbacks=[rlrop, es_cb])
    return history, model


# train, predict for each step after the init interval, 
# do not go further than  we can becase we have no further data
def loop(model,init_interval, x, df, options, train=True, verbose=1):
    results = []
    init_interval = max(init_interval, options["lookback_interval"])
    for i in range(init_interval, len(x)-options["lookback_interval"]):
        # print the current timetamp
        ts, pred = predict_one_step(model, x, i, options["lookback_interval"], options["features"])
       
        actual_close = df["close"].iloc[i]
        results.append([ts, pred, actual_close])
    
        if verbose == 1:
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
    "filepath": "data/stocks/SPY.csv",
    "lookback_interval":40,
    "lookahead_interval": 2,
    "features" : "auto",
    "out": 1
}

training_fraction = 0.4

df = load_dataset(options["filepath"])

df_orig = df.copy()

x, y, close_scaler = setup_dataset(df, 40, options["out"])
init_interval = int(len(x) * training_fraction)
# set features correctly
if options["features"] == "auto":
    options["features"] = x.shape[-1]


model = setup_model(**options)


rlrop = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_delta=0.001)
es_cb = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=0, restore_best_weights=True)

hist, model = train_init(model, init_interval, x, y, cbs=[rlrop, es_cb])

visualize_training_results(hist)


#%%

# make a in the loop train predict loop over the rest of the data instead
results, model = loop(model, init_interval, x, df_orig, options, train=True)

#%%


# run a full simulation run from the start:
# 40 -> because we currently use 40 timesteps. this will be checked in the function anyways
#   here just added for clarity
try:
    results
except NameError:
   results, model = loop(model, 40, x, df_orig, options, train=False, verbose=0)


plot_result(results)




#%%

def decision_model(act, pred):
    pass

predictions
actual.loc[pd.date_range("2016-06-13", "2016-06-17")]

((actual.shift(1) - predictions)/actual.shift(1)).dropna()


# first job: order into classes instead of regression
# second job: create useful prediction / value function for comparability
# fourth job: add additional features (currently only one instrument)
















