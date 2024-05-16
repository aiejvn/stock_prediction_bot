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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import seaborn as sns


# Preprocess all data




initial = False # True if we have no data - tweaks may be needed
scaler = MinMaxScaler(feature_range=(-1, 1))
x_scaler = StandardScaler()
y_scaler = StandardScaler()

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100 
num_tune_epochs = 3 

# True for Train, False for Load
train_or_load_original = True # separate modifier because original model's performance is very unreliable
train_or_load_new = True
train_or_load = True
train_or_load_gru = True





def save_data(data, date = None):
    if not date:
        date = f"{str(datetime.now()).replace(':','.')}.csv"
    os.makedirs("data", exist_ok=True)
    contents = os.listdir("data")
    for item in contents:
        if date[:10] in item:
            os.remove("data/" + item)
    if f"data/{date}" not in os.listdir("data"):
        data.to_csv("data/" + date, index=False)
        
def read_data(filepath, sort_column = None):
    data = pd.read_csv(filepath)
    if sort_column:
        data = data.sort_values(sort_column)
    data.head()
    price = data[['Close']] 
    price['Close'] = price['Close'].values.reshape(-1,1)
    return data, price





data = pd.read_csv("relevant-stock-data/AMZN Historical Data Fixed.csv")
data = data.iloc[::-1]
data = data[['Date', 'Open', 'High', 'Low', 'Price', 'Vol.']]
data = data.rename(columns={'Price' : 'Close', 'Vol.' : 'Volume'})
data['Name'] = ['AMZN']*len(data['Date'])

# iloc for integer based assignment of multiple cells, iat for integer based assignment on a single cell
for i in range(len(data['Volume'])): 
    if 'M' in data['Volume'].iat[i]:
        data['Volume'].iat[i] = str(int(float(data['Volume'].iat[i][:-1]) * 1e6))
    if 'B' in data['Volume'].iat[i]:
        data['Volume'].iat[i] = str(int(float(data['Volume'].iat[i][:-1]) * 1e9))

price = data[['Close']]
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
print(price.iloc[-5:])
print(data.iloc[-5:])





def split_data(data, stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    print(data.shape)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    x_train[:,:,0] = x_scaler.fit_transform(x_train[:,:,0])
    y_train = y_scaler.fit_transform(data[:train_set_size,-1,:])
    
    x_test = data[train_set_size:,:-1]
    x_test[:,:,0] = x_scaler.transform(x_test[:,:,0])
    y_test = y_scaler.transform(data[train_set_size:,-1,:])
    
    return [x_train, y_train, x_test, y_test]





# 0 for the original, 1 for our new lstm and gru, 2 for the expanded lstm 
# expected to have many models - lists preferred over dictionaries
x_trains = []
y_trains_lstm = []
x_tests = []
y_tests_lstm = []





lookback = 20 # choose sequence length

x_train, y_train, x_test, y_test = split_data(data, price, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)





# Convert arrays to Pytorch arrays using .from_numpy()
# Set type to torch tensors using .type(torch.Tensor)
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)

x_trains.append(x_train)
x_tests.append(x_test)
y_trains_lstm.append(y_train_lstm)
y_tests_lstm.append(y_test_lstm)





if not initial:
    print("data/" + os.listdir("data")[-1]) # do we save the wrong data?
    data = pd.read_csv("data/" + os.listdir("data")[-1])


# Automatically Updating Data




cur_time = int(str(datetime.now())[11:13])
print(cur_time) # the current hour
most_recent_date = data['Date'].iloc[-1]
print(most_recent_date) # broken lol
print(str(datetime.now())[:10])





def get_updated_data():
    count_today = cur_time >= 16 # has the market closed yet? T if yes, F if no.
    url = "https://www.investing.com/equities/amazon-com-inc-historical-data"
    page = requests.get(url)
        
    soup = BeautifulSoup(page.text, 'html.parser')
        
    # We need Open, High, Low, Close, Volume, Name
    all_date_class = "freeze-column-w-1 w-full overflow-x-auto text-xs leading-4"
    table = soup.find('table', {'class': all_date_class})
    rows = table.find_all('tr')
    
    table_data = []
    for row in rows:
        row_data = []
        for cell in row.find_all(['td']):
            row_data.append(cell.text)
        if row_data and count_today:
            
            for i in range(len(row_data)):
                if 'M' in row_data[i]:
                    row_data[i] = str(int(float(row_data[i][:-1]) * 1e6)) # all data values are rounded so we round too
            if most_recent_date in row_data:
                break # we've caught up
            table_data.append(row_data)
        else:
            count_today = True

    # print(table_data)
    return table_data

get_updated_data()





date_updated = most_recent_date[-4:] + "-" + most_recent_date[:2] + "-" + most_recent_date[3:5] 
need_update = not date_updated == str(datetime.now())[:10] and cur_time >= 16
print(need_update)

if need_update:
    new_data = get_updated_data()[::-1] # newest dates are returned first
    start_index = len(data['Date'])
        
    for i in range(len(new_data)): 
        new_values = new_data[i] # comes as date, price (close), open, high, low, vol, change
    
        # new_values = get_updated_data()[0] 
        print(pd.DataFrame(new_values).head())
        
        # read, write, then save the new values inside our most recent table
        data.loc[start_index + i] = [new_values[0], new_values[2], new_values[3], new_values[4], new_values[1], new_values[5], "AMZN"] 
    
    save_data(data)





updated_data, updated_price = read_data('data/' + os.listdir("data")[-1])
print('data/' + os.listdir("data")[-1])
print(updated_data.iloc[-5:])
print(updated_price.iloc[-5:])





if('2024-04-26' not in str(datetime.now())): # initial date
    if need_update:
        save_data(data)  
else:
    save_data('2024-04-26 00.00.000000')
print(str(datetime.now())[:10])





lookback = 20 # choose sequence length
x_train, y_train, x_test, y_test = split_data(updated_data, updated_price, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

# print(x_train[0])
# print(y_train[0])
# print(x_test[0])
# print(y_test[0])





x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)

x_trains.append(x_train)
x_tests.append(x_test)
y_trains_lstm.append(y_train_lstm)
y_tests_lstm.append(y_test_lstm)





volume_scaler = MinMaxScaler(feature_range=(-1, 1)) # The volumes are in the millions and would mess up our scaler
column_scaler = StandardScaler()





# Expanded data
# Scale and transform the other columns ONLY HERE so that we don't save the scaled values
def split_data_expanded(data, stock, lookback):
    data_raw = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_raw = stock.to_numpy() # convert to numpy array
    x = []
    y = []
    
    data_raw = data_raw.to_numpy()
    
    # create all possible sequences of length seq_len
    for index in range(len(stock_raw) - lookback): 
        x.append(data_raw[index: index + lookback])
        y.append(stock_raw[index: index + lookback])
    
    x = np.array(x)
    y = np.array(y)
    
    print(x.shape)
    print(y.shape)
    
    test_set_size = int(np.round(0.2*y.shape[0])) # 80-20 train-test split
    train_set_size = y.shape[0] - (test_set_size)
    
    # scale all values in data_raw
    for i in range(x.shape[2]): 
        x[:,:,i] = column_scaler.fit_transform(x[:,:,i])
        
    # x_train = y[:train_set_size,:-1,:]
    x_train = x[:train_set_size,:-1,:]
    y_train = y_scaler.fit_transform(y[:train_set_size,-1,:])
    
    # x_test = y[train_set_size:,:-1]
    x_test = x[train_set_size:,:-1,:]
    y_test = y_scaler.transform(y[train_set_size:,-1,:])
    
    return [x_train, y_train, x_test, y_test]

# Test code
test = True
if test:
    test_data, test_stock = read_data("data/" + os.listdir("data")[-1])
    print(test_data.iloc[-5:])
    x_train, y_train, x_test, y_test = split_data_expanded(test_data, test_stock, 20)
    print('x_train.shape = ',x_train.shape)
    print('y_train.shape = ',y_train.shape)
    print('x_test.shape = ',x_test.shape)
    print('y_test.shape = ',y_test.shape)





# train, test key is 2
x_train, y_train, x_test, y_test = split_data_expanded(updated_data, updated_price, lookback)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)





x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)

x_trains.append(x_train)
x_tests.append(x_test)
y_trains_lstm.append(y_train_lstm)
y_tests_lstm.append(y_test_lstm)


# Train models - one on the provided data, rest on the collected data

# Also, fine-tune the models' hyperparameters 




model_dir = "models"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.MSELoss(reduction='mean')





class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out





class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out





original_model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
original_optimiser = torch.optim.Adam(original_model.parameters(), lr=0.01)

if not train_or_load_original:
    # load original model
    loaded_state = torch.load(model_dir + "/original3.pt", map_location=device)
    original_model.load_state_dict(loaded_state['model'])
    original_optimiser.load_state_dict(loaded_state['opt'])





def train_model(model, optimiser, data_key, num_epochs=100):
    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    for t in range(num_epochs):
        y_train_pred = model(x_trains[data_key])

        loss = criterion(y_train_pred, y_trains_lstm[data_key])
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))
    return y_train_pred, hist





if need_update and train_or_load_original:
    y_train_pred, hist = train_model(original_model, original_optimiser, 0, num_epochs=num_epochs)





def graph_results(y_train_pred, hist, data_key, model='LSTM'):
    predict = pd.DataFrame(y_scaler.inverse_transform(y_train_pred.detach().numpy())) # un-normalize data
    original = pd.DataFrame(y_scaler.inverse_transform(y_trains_lstm[data_key].detach().numpy())) # un-normalzie data
    
    sns.set_style("darkgrid")    

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    
    if model == 'LSTM':
        ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
    else:
        ax = sns.lineplot(x = predict.index, y = predict[0], label=f"Training Prediction ({model})", color='tomato')
    ax.set_title('Stock price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (USD)", size = 14)
    ax.set_xticklabels('', size=10)


    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Training Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)
    
    plt.show()





if need_update and train_or_load_original:
    graph_results(y_train_pred, hist, 0)





new_model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
new_optimiser = torch.optim.Adam(new_model.parameters(), lr=0.01)

if not train_or_load_new:
    # load a previous model
    possible_models = []
    for i in os.listdir(model_dir):
        if 'stock-only-lstm' in i:
            possible_models.append(i)
            print(i)
    
    loaded_state = torch.load(model_dir + "/" + possible_models[-1], map_location=device)
    new_model.load_state_dict(loaded_state['model'])
    new_optimiser.load_state_dict(loaded_state['opt'])





if not train_or_load_new:
    if need_update:
        y_pred, hist = train_model(new_model, new_optimiser, 1, num_epochs=num_tune_epochs)
else:
    y_pred, hist = train_model(new_model, new_optimiser, 1, num_epochs=num_epochs)





if need_update or train_or_load_new:
    graph_results(y_pred, hist, 1)





# Train a model using all traits - not just stock price
# 5 important traits - open, high, low, close, volume
# data key is 2
expanded_model = LSTM(input_dim=5, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
expanded_optimiser = torch.optim.Adam(expanded_model.parameters(), lr=0.01)

if not train_or_load:
    # load a previous model
    possible_models = []
    for i in os.listdir(model_dir):
        if 'expanded-lstm' in i:
            possible_models.append(i)
            print(i)
    
    loaded_state = torch.load(model_dir + "/" + possible_models[-1], map_location=device)
    expanded_model.load_state_dict(loaded_state['model'])
    expanded_optimiser.load_state_dict(loaded_state['opt'])

if not train_or_load:
    if need_update: 
        y_pred, hist = train_model(expanded_model, expanded_optimiser, 2, num_epochs=num_tune_epochs)
else:
    y_pred, hist = train_model(expanded_model, expanded_optimiser, 2, num_epochs=num_epochs)
    





if need_update or train_or_load:
    graph_results(y_pred, hist, 2)





# use data key of 1 - same as the new lstm model
new_gru = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
new_gru_optimiser = torch.optim.Adam(new_gru.parameters(), lr=0.01)

if not train_or_load_gru:
    # load a previous model
    possible_models = []
    for i in os.listdir(model_dir):
        if 'stock-only-gru' in i:
            possible_models.append(i)
            print(i)
    
    loaded_state = torch.load(model_dir + "/" + possible_models[-1], map_location=device)
    new_gru.load_state_dict(loaded_state['model'])
    new_gru_optimiser.load_state_dict(loaded_state['opt'])    
    if need_update:
        y_pred, hist = train_model(new_gru, new_gru_optimiser, 1, num_epochs=num_tune_epochs)
else:
    y_pred, hist = train_model(new_gru, new_gru_optimiser, 1, num_epochs=num_epochs)





if need_update or train_or_load_gru:
    graph_results(y_pred, hist, 1, model='GRU')


# Make Predictions using Models 

# Original and new LSTM model just do regression on the stock themselves \
# Expanded model uses open, high, low, close, volume to regress stock




def buy_or_sell(is_buy, start, stop, prices, days):
    if is_buy:
        min_price = min(prices[start:stop])
        for j in range(start, stop):
            if prices[j] != min_price:
                days[j] = None
    else:    
        max_sell = max(prices[start:stop])
        for j in range(start, stop):
            if prices[j] != max_sell:
                days[j] = None

def maximize_profit(prices):
    cur_hold, cur_not_hold = -float('inf'), 0
    days = [None] * len(prices)

    for i in range(len(prices)):
        stock_price = prices[i]
        prev_hold, prev_not_hold = cur_hold, cur_not_hold
        
        cur_hold = max(prev_hold, prev_not_hold - stock_price)
        if prev_hold < prev_not_hold - stock_price: # if it is more worth to buy
            days[i] = "buy"
            
        
        cur_not_hold = max(prev_not_hold, prev_hold + stock_price)
        if prev_not_hold < prev_hold + stock_price: # if it is more worth to sell
            days[i] = "sell" 

    is_buy = False
    days_to_consider = []   

    for i in range(len(days)):
        if (len(days_to_consider) > 0 and (days[i] == "buy") != is_buy):
            start, stop = days_to_consider[0], days_to_consider[-1] + 1
            buy_or_sell(is_buy, start, stop, prices, days)
            days_to_consider = []
        days_to_consider.append(i) 
        is_buy = days[i] == 'buy'
        
    if len(days_to_consider) > 0:
        buy_or_sell(is_buy, days_to_consider[0], days_to_consider[-1] + 1, prices, days) 
        
    return cur_not_hold, days





# predict an entire week 
# NOT compatible with the expanded LSTM - do not use!!
def predict_entire_week(model, x_pred):
    cur_price = int(updated_data['Close'].values[-1])
    results = [cur_price]
    today = datetime.now()
    days_left = max(0, 5-today.weekday()) if today.weekday() < 6 else 5
    for i in range(days_left):
        pred = model(x_pred)
        x_pred = torch.cat((x_pred, pred.reshape(1,1,1)), dim=1)
        results.append(scaler.inverse_transform(np.array(pred.detach()))[0,0])
        
    print(results)
        
    cur_not_hold, days = maximize_profit(results)
           
    return results, days, cur_not_hold

test = True
if test:
    x_pred = []
    x_pred.append(updated_price.values[-20:].astype(np.float32))
    x_pred = torch.tensor(x_pred) 
    print(predict_entire_week(new_model, x_pred))





x_pred = []
x_pred.append(updated_price.values[-20:].astype(np.float32))
x_pred = torch.tensor(x_pred) 
output_dir = "data"
date = str(datetime.now()).replace(":", ".")

next_price_original =  y_scaler.inverse_transform(np.array(original_model(x_pred).detach()))[0,0] 
print("Original LSTM:", next_price_original)

next_price_new = y_scaler.inverse_transform(np.array(new_model(x_pred).detach()))[0,0]
print("Updated LSTM:", next_price_new)

next_price_gru = y_scaler.inverse_transform(np.array(new_gru(x_pred).detach()))[0,0]
print("Updated GRU:", next_price_gru)

data = updated_data.iloc[-20:,1:6]
n = len(data.iloc[0,:])
for i in range(n - 1):
    data.iloc[:,i] = scaler.fit_transform(data.iloc[:,i].values.reshape(-1,1))
data.iloc[:,n-1] = volume_scaler.fit_transform(data.iloc[:, n-1].values.reshape(-1,1))
x_pred = []
x_pred.append(data.values[-20:].astype(np.float32)) # giving all values breaks it
x_pred = torch.tensor(x_pred)
next_price_expanded = scaler.inverse_transform(np.array(expanded_model(x_pred).detach()))[0,0]
print("Expanded and Updated LSTM:", next_price_expanded)

with open("predictions/original_predictions " + date + ".txt", "w") as file:
    file.write(str(next_price_original)) # original is predicting stock values too low - needs a fix?
with open("predictions/new_predictions " + date + ".txt", "w") as file:
    file.write(str(next_price_new)) 
with open("predictions/new_and_expanded_predictions " + date + ".txt", "w") as file:
    file.write(str(next_price_expanded))





folder_dir = "models/"
files = os.listdir(folder_dir)
while len(os.listdir(folder_dir)) > 100:
    os.remove(folder_dir + files.pop(0))

filename = str(datetime.now()).replace(":", ".") + ".pt"

if train_or_load_original:
    torch.save({
        "model":original_model.state_dict(),
        "opt":original_optimiser.state_dict()
    }, folder_dir + "/original3.pt")  
torch.save({ 
    "model": new_model.state_dict(),
    "opt": new_optimiser.state_dict()
}, folder_dir + "/stock-only-lstm " + filename)
torch.save({ 
    "model": expanded_model.state_dict(),
    "opt": expanded_optimiser.state_dict()
}, folder_dir + "/expanded-lstm " + filename)
torch.save({ 
    "model": new_gru.state_dict(),
    "opt": new_gru_optimiser.state_dict()
}, folder_dir + "/stock-only-gru " + filename)


#  Verifying Accuracy of Models 




def verify(model, key):
    x = x_tests[key]
    y = y_tests_lstm[key]
    pred = model(x)
    mse = criterion(pred, y) # function only uses lists or numpy arrays
    return mse.item()

print("MSE of Original LSTM", verify(original_model, 0))
print("MSE of New LSTM", verify(new_model, 1))
print("MSE of New GRU", verify(new_gru, 1))
print("MSE of Expanded Model", verify(expanded_model, 2))

