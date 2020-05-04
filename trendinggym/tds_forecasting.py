# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:33:14 2020

@author: Richard

recreating the stock forecasting tds article
# https://towardsdatascience.com/using-deep-learning-ai-to-predict-the-stock-market-9399cf15a312
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

def validater(n_per_in, n_per_out):
    """
    Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
    Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
    """
    
    # Creating an empty DF to store the predictions
    predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]])

    for i in range(n_per_in, len(df)-n_per_in, n_per_out):
        # Creating rolling intervals to predict off of
        x = df[-i - n_per_in:-i]

        # Predicting using rolling intervals
        yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))

        # Transforming values back to their normal prices
        yhat = close_scaler.inverse_transform(yhat)[0]

        # DF to store the values and append later, frequency uses business days
        pred_df = pd.DataFrame(yhat, 
                               index=pd.date_range(start=x.index[-1], 
                                                   periods=len(yhat), 
                                                   freq="B"),
                               columns=[x.columns[0]])

        # Updating the predictions DF
        predictions.update(pred_df)
        
    return predictions



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

# Loading in the Data
df = pd.read_csv("data/stocks/SPY.csv", index_col=0)

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

# Only using the last 1000 days of data to get a more accurate representation of the current market climate
# df = df.tail(1000)



## Scaling

# Scale fitting the close prices separately for inverse_transformations purposes later
close_scaler = RobustScaler()

close_scaler.fit(df[['close']])

# Normalizing/Scaling the DF
scaler = RobustScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

#%%
## Creating the NN


def setup_model(lookback_interval, lookahead_interval, features, out):
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


def train_init(model, init_interval, x, y):
    #baseline training with x%
    x_t = x[0:init_interval,:,:]
    y_t = y[0:init_interval]
    
    hist = model.fit(x_t, y_t, 
                    epochs=25, batch_size=64, validation_split=0.1,
                     callbacks=[rlrop, es_cb])

    return hist, model



def predict_one_step(model, df, iteration, lookback_interval, features):
    x = df.iloc[iteration-lookback_interval:iteration]

    # Predicting using rolling intervals
    yhat = model.predict(np.array(x).reshape(1, lookback_interval, features))

    # Transforming values back to their normal prices
    yhat = close_scaler.inverse_transform(yhat)[0]

    # DF to store the values and append later, frequency uses business days
    return (df.index[iteration+1], yhat[-1])
    

def setup_dataset(df, input_shape, output_shape):
    # input_shape, output_shape = 30, 1
    # Splitting the data into appropriate sequences
    X, y = split_sequence(df.to_numpy(), input_shape, output_shape)
    
        
    #split in train and validation data
    # X_train, X_val,y_train, y_val = train_test_split(X,y, shuffle=False)
    # permutate the train data 
    
    """
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    """
    return X,y 


def retrain(model, i, replay_buffer, x, y):
    if replay_buffer == 0:
        start = 0
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


"""
    TRAIN THE MODEL!
"""


options = {
    "lookback_interval":40,
    "lookahead_interval": 2,
    "features" : df.shape[1],
    "out": 1
}

testmodel = setup_model(**options)
x,y = setup_dataset(df, 40, 1)



init_interval = int(len(x)*0.4)

rlrop = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_delta=0.001)
es_cb = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=0, restore_best_weights=True)

hist, model = train_init(model, init_interval, x, y)



results = []
# train, predict for each step after the init interval, 
# do not go further than  we can becase we have no further data

i=init_interval+1
for i in range(init_interval, len(x)-options["lookback_interval"]):
    # print the current timetamp
    ts, pred = predict_one_step(model, df, i, options["lookback_interval"], options["features"])
   
    actual_close = close_scaler.inverse_transform(df["close"].iloc[i+1].reshape(-1,1))[0][0]
    results.append([ts, pred, actual_close])

    print(f"predicting: {df.index[i]}:\t{pred:.4f} | {actual_close:.4f}")
    
    _, model = retrain(model, i, replay_buffer=512, x=x, y=y)




"""
    PLOT RESULTS
"""


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


plt.axvspan(results.index.values[init_interval],
            results.index.values[-1],
            alpha=0.2
            )

plt.title(f"Predicted vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
plt.show()



"""
    OLD PLOTTING
"""
def plot_results_array(results):
    predictions = pd.DataFrame(val, index=ts, columns=["close"]).sort_index()
    results = pd.DataFrame(results, index=0, columns=["pred", "act"])

    

#
# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(df[["close"]]), 
                      index=df.index, 
                      columns=[df.columns[0]])

# Creating an empty DF to store the predictions
predictions = pd.DataFrame(index=df.index, columns=[df.columns[0]])
ts, val = [], []
iterator = list(range(n_per_in, len(df)-n_per_in, 1))
i = iterator[1]
for i in iterator:
    # Creating rolling intervals to predict off of
    x = df[-i - n_per_in:-i]

    # Predicting using rolling intervals
    yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))

    # Transforming values back to their normal prices
    yhat = close_scaler.inverse_transform(yhat)[0]

    # DF to store the values and append later, frequency uses business days
    ts.append(pd.date_range(start=x.index[-1], 
                        periods=len(yhat), 
                        freq="B")[-1])
    
    val.append(yhat[-1])
    

    # Updating the predictions DF
    # predictions.update(pred_df)

predictions = pd.DataFrame(val, index=ts, columns=["close"]).sort_index()


# Printing the RMSE
print("RMSE:", val_rmse(actual, predictions))

number = 20
def plot_abschnitt(number):
    from pandas.tseries.offsets import BDay
    start = actual.index.values[number]
    real = predictions.index.values[number + n_per_in + 1]

    fig = plt.figure()
    act = actual[start:real-BDay(1)]
    pred = predictions[real].to_frame().transpose()
    plt.plot(act.index, act.values, label='actual')    
    plt.plot(pred.index, pred.values, 'r.', label='Predicted')
    


s_ts = max(actual.index.values[0], predictions.index.values[0])
e_ts = min(actual.index.values[-1], predictions.index.values[-1])
# Plotting
plt.figure(figsize=(16,6))


# Plotting the actual values
plt.plot(actual[s_ts:e_ts], label='Actual')
# Plotting those predictions
plt.plot(predictions[s_ts:e_ts], 'r-', label='Predicted')

plt.axvspan(predictions.index.values[len(X_train)],
            e_ts,
            alpha=0.2
            )

plt.title(f"Predicted vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
plt.show()





#%%

def decision_model(act, pred):
    pass

predictions
actual.loc[pd.date_range("2016-06-13", "2016-06-17")]

((actual.shift(1) - predictions)/actual.shift(1)).dropna()


# first job: order into classes instead of regression
# second job: create useful prediction / value function for comparability
# third job: train on time x, predict for x+1, train on time x+1, predict for x+2 ...
# fourght job: add additional features (currently only one instrument)



#%%

ticker = "SPY"
spy = get_stock_data(ticker, "2011-01-01", pd.Timestamp.now())














