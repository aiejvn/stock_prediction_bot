import pandas as pd
import numpy as np
from datetime import datetime

data = pd.read_csv("relevant-stock-data/AMZN Historical Data Fixed.csv")
data = data.iloc[::-1]
data = data[['Date', 'Open', 'High', 'Low', 'Price', 'Vol.']]
data = data.rename(columns={'Price' : 'Close', 'Vol.' : 'Volume'})
data['Name'] = ['AMZN']*len(data['Date'])

data.head()

for i in range(len(data['Volume'])):
    if 'M' in data['Volume'].iloc[i]:
        data['Volume'].iloc[i] = str(int(float(data['Volume'].iloc[i][:-1]) * 1e6))
    if 'B' in data['Volume'].iloc[i]:
        data['Volume'].iloc[i] = str(int(float(data['Volume'].iloc[i][:-1]) * 1e9))

date = f"{str(datetime.now()).replace(':','.')}.csv"
data.to_csv("data/" + date, index=False)