# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:56:23 2020

@author: Richard
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout

def split_sequence(seq, n_steps_in, n_steps_out):
    """
    Splits the multivariate time sequence
    """
    seq = seq.values
    # Creating a list for both variables
    X, y = [], []
    
    for i in range(len(seq)):
        
        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + n_steps_out
        
        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(seq):
            break
        
        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)
  
         
def visualize_training_results(results):
    """
    Plots the loss and accuracy for the training and testing data
    """
    history = results.history
    fig, ax = plt.subplots(2,1, figsize=(16,5))
    ax1 = ax[0]
    ax1.plot(history['loss'])
    
    if "val_loss" in history:
        ax1.plot(history['val_loss'])
        ax1.legend(['val_loss', 'loss'])
    else:
        ax1.legend(['loss'])
        
    ax1.title.set_text('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    
    ax2 = ax[1]
    if "val_acc" in history:
        history["val_accuracy"] = history["val_acc"]
    
    if "acc" in history:
        history["accuracy"] = history["acc"]
    
    ax2.plot(history['accuracy'])
    
    if "val_accuracy" in history:
        ax2.plot(history['val_accuracy'])
        ax2.legend(['val_accuracy', 'accuracy'])
    else:
        ax2.legend(['accuracy'])
        
    ax2.title.set_text('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    plt.show()
    
          
def val_rmse(df1, df2):
    """
    Calculates the root mean square error between the two Dataframes
    """
    df = df1.copy()
    
    # Adding a new column with the closing prices from the second DF
    df['close2'] = df2.close
    
    # Dropping the NaN values
    df.dropna(inplace=True)
    
    # Adding another column containing the difference between the two DFs' closing prices
    df['diff'] = df.close - df.close2
    
    # Squaring the difference and getting the mean
    rms = (df[['diff']]**2).mean()
    
    # Returning the sqaure root of the root mean square
    return float(np.sqrt(rms))
  