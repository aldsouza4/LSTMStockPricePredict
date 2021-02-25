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


class PredictPriceNN(object):

    def __init__(self, ticker):
        self.df = pd.DataFrame()
        self.df = wb.DataReader(ticker, data_source='yahoo', start='2011-1-1')['Adj Close']
        self.df = self.df.to_frame()
        self.df = self.df.reset_index()
        self.df.drop(['Date'], axis=1, inplace=True)

    def scale_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df = self.scaler.fit_transform(np.array(self.df).reshape(-1, 1))
        return self.df

    def make_data(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.time_step - 1):
            dataX.append(dataset[i:(i + self.time_step), 0])
            dataY.append(dataset[i + self.time_step, 0])
        return np.array(dataX), np.array(dataY)

    def clean_data(self):
        self.scale_data()
        self.time_step = 240
        self.X_train, self.y_train = self.make_data(self.df[:])
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)


    def LSTM_model(self, epochs=2, batchsize=64):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.time_step, 1)))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batchsize, verbose=1)


    def make_predict_dataset(self, plot=False, model="LSTM", epochs=2, batchsize=64):
        self.clean_data()

        if model == "LSTM":
            self.LSTM_model(epochs, batchsize)

        x_input = self.df[-self.time_step:].reshape(1, -1)
        temp_input = x_input[0].tolist()
        lst_output = []
        predict_further = 240
        i = 0
        while i < predict_further:

            if len(temp_input) > self.time_step:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, self.time_step, 1))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1

            else:
                x_input = x_input.reshape((1, self.time_step, 1))
                yhat = self.model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1

        if plot:
            day_new = np.arange(1, self.time_step + 1)
            day_pred = np.arange(self.time_step + 1, self.time_step + predict_further + 1)

            plt.plot(day_new, self.scaler.inverse_transform(self.df[-self.time_step:]))
            plt.plot(day_pred, self.scaler.inverse_transform(lst_output))
            plt.show()


        return lst_output



if __name__ == '__main__':

    t = PredictPriceNN("HDFCBANK.NS")
    # print(t.make_predict_dataset())
    print(t.make_predict_dataset(plot=True))



