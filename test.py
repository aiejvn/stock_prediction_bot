import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from sklearn.preprocessing import MinMaxScaler
import time
import seaborn as sns

scaler = MinMaxScaler(feature_range=(-1, 1))

def read_data(filepath, sort_column = None):
    data = pd.read_csv(filepath)
    if sort_column:
        data = data.sort_values(sort_column)
    data.head()
    price = data[['Close']]
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
    return data, price

def split_data(data, stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

data, price = read_data('relevant-stock-data\AMZN_2006-01-01_to_2018-01-01.csv', 'Date')
print(price.info())
print(data.head())

lookback = 20 # choose sequence length
x_train, y_train, x_test, y_test = split_data(data, price, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

print(x_train[0])
print(y_train[0])
print(x_test[0])
print(y_test[0])