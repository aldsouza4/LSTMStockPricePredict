import pandas as pd
import pandas_datareader as wb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from numpy import array


def makeCSV(ticker):
    df = pd.DataFrame()
    df = wb.DataReader(ticker, data_source='yahoo', start='2011-1-1')['Adj Close']
    df = df.to_frame()
    df = df.reset_index()
    df.to_csv('stock')


makeCSV("HDFCBANK.NS")

df = pd.read_csv('stock')
df.drop(['Unnamed: 0', 'Date'], axis=1, inplace=True)


scaler = MinMaxScaler(feature_range=(0, 1))

df = scaler.fit_transform(np.array(df).reshape(-1, 1))

train_size = int(len(df) * 0.80)
test_size = (len(df) - train_size)

train_data, test_data = df[:train_size], df[train_size:]




def make_datasets(dataset, time_steps=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_steps-1):
        dataX.append(dataset[i:(i+time_steps), 0])
        dataY.append(dataset[i + time_steps, 0])

    return np.array(dataX), np.array(dataY)


time_step = 240

X_train, y_train = make_datasets(train_data, time_step)
X_test, y_test = make_datasets(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


model=Sequential()
model.add(LSTM(50, return_sequences=True,input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')



model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64, verbose=1)

x_input = test_data[-time_step:].reshape(1, -1)
temp_input = x_input[0].tolist()

lst_output = []
n_steps = time_step
predict_further = 240

i = 0
while i < predict_further:

    if len(temp_input) > n_steps:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1

    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1



day_new = np.arange(1, n_steps + 1)
day_pred = np.arange(n_steps + 1, n_steps + predict_further + 1)

plt.plot(day_new,scaler.inverse_transform(df[-time_step:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.show()