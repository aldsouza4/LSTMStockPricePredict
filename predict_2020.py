import pandas as pd
import pandas_datareader as wb
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from datetime import datetime
from dateutil.relativedelta import relativedelta

sns.set_style("darkgrid")


class PredictPriceNN(object):

    def __init__(self, ticker):
        """

        :param ticker: enter the ticker symbol of the stock
        """
        self.ticker = ticker
        self.df = pd.DataFrame()
        self.df = wb.DataReader(self.ticker, data_source='yahoo', start='2011-1-1')['Adj Close']
        self.df = self.df.to_frame()
        self.df = self.df.reset_index()
        self.df = self.df.iloc[:-240]
        self.full_df = self.df
        self.full_df_w_date = self.df
        self.df.drop(['Date'], axis=1, inplace=True)
        # print(self.df)

    def scale_data(self):
        """
        method to scale the data ( as scaled data works better in LSTM models)
        :return: returns the scaled data
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.df = self.scaler.fit_transform(np.array(self.df).reshape(-1, 1))
        return self.df

    def make_data(self, dataset):
        """
        making feature dataset using time_step
        time_steps will be the columns that will be passed to the model ( or X dataset )
        :param dataset: takes in the scaled data
        :return: return np array of X data and y
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - self.time_step - 1):
            dataX.append(dataset[i:(i + self.time_step), 0])
            dataY.append(dataset[i + self.time_step, 0])
        return np.array(dataX), np.array(dataY)

    def clean_data(self):
        """
        prepares the data to be used to fit the model

        :return:
        """
        self.scale_data()
        self.time_step = 20
        self.X_train, self.y_train = self.make_data(self.df[:])
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)

    def LSTM_model(self, epochs=1, batchsize=128):
        """
        Tensorflow LSTM model
        :param epochs: epochs for model to train
        :param batchsize: batch size for model
        :return:
        """
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.time_step, 1)))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batchsize, verbose=1)

    def make_predict_dataset(self, plot=False, model="LSTM", epochs=1, batchsize=128, predict_no_yrs=1,
                             plot_with_date=False):
        """

        :param plot_with_date:
        :param plot: True if Plot
        :param model: input model to be used
        :param epochs: epochs for model to train
        :param batchsize: batch size for model
        :return:
        """
        self.clean_data()

        if model == "LSTM":
            self.LSTM_model(epochs, batchsize)

        x_input = self.df[-self.time_step:].reshape(1, -1)
        temp_input = x_input[0].tolist()
        lst_output = []
        self.predict_further = predict_no_yrs * 240

        i = 0
        while i < self.predict_further:

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

        stock_price = self.scaler.inverse_transform(lst_output)[0][0]

        if plot:
            day_new = np.arange(1, self.time_step + 1)
            day_pred = np.arange(self.time_step + 1, self.time_step + self.predict_further + 1)

            plt.figure(figsize=(15, 8))
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.title("{} DL - LSTM".format(self.ticker), fontsize=18)

            sns.lineplot(x=day_new, y=self.scaler.inverse_transform(self.df[-self.time_step:]).flatten().tolist())
            sns.lineplot(x=day_pred, y=self.scaler.inverse_transform(lst_output).flatten().tolist())
            sns.lineplot(x=[self.time_step, self.time_step + 1],
                         y=[self.full_df.iloc[-1]['Adj Close'], self.scaler.inverse_transform(lst_output)[0][0]])

            plt.tight_layout()
            plt.show()

        elif plot_with_date:
            start_date = (datetime.now()) - relativedelta(years=5)
            self.start_input = "{0}-{1}-{2}".format(start_date.year, start_date.month, start_date.day)
            end_date = datetime.now() - relativedelta(years=1)
            self.end_input = "{0}-{1}-{2}".format(end_date.year, end_date.month, end_date.day)

            stock_price_data = \
                wb.DataReader(self.ticker, data_source='yahoo', start=self.start_input, end=self.end_input)[
                    'Adj Close']
            stock_price_data = stock_price_data.to_frame()
            stock_price_data = stock_price_data.reset_index()
            p_date = datetime.now() + relativedelta(years=predict_no_yrs)
            stock_price_data['Date'] = stock_price_data['Date'].apply(lambda x: x.date())
            predict_data = pd.DataFrame(columns=['Date', 'Adj Close'])
            predict_data.loc[0] = [stock_price_data.iloc[-1]['Date'], stock_price_data.iloc[-1]['Adj Close']]
            predict_data.loc[1] = [p_date.date(), stock_price]

            plt.figure(figsize=(15, 8))
            sns.set_style("darkgrid")
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.title("{} DL - LSTM".format(self.ticker), fontsize=18)
            plt.tight_layout()
            plt.plot_date(x=predict_data['Date'], y=predict_data['Adj Close'], linestyle='solid', marker=None)
            plt.plot_date(x=stock_price_data['Date'], y=stock_price_data['Adj Close'], linestyle='solid', marker=None)
            plt.legend(labels=["Expectation", 'Declared'])
            plt.show()

        return self.scaler.inverse_transform(lst_output)[0][0]


if __name__ == '__main__':
    t = PredictPriceNN("wipro.NS")
    print(t.make_predict_dataset(plot_with_date=True))
