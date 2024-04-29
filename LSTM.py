import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
#from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

class MyLSTM():
    def __init__(self, df, show_data=False, split=0.8):
        self.df = df
        self.show_data = show_data
        self.split = split
        
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_split()

        self.model = self.create_model()
        
        self.history = None
    
    def data_split(self):
        n = len(self.df)
        X = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=(1-self.split), random_state=42)

        # Reshape input to be 3D [samples, timesteps, features] for LSTM
        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

        return self.X_train, self.y_train, self.X_test, self.y_test
        
    def create_model(self):
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def fit_model(self, epochs=20, batch_size=70):
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test), verbose=2, shuffle=False)

        if self.show_data:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            plt.show()
        
        return history

    def predict(self, scaler : MinMaxScaler):
        y_hat = self.model.predict(self.X_test)
        y_hat = y_hat.reshape(-1, 1)
        scaler.inverse_transform(y_hat)

        # Invert scaling for forecast
        print(y_hat)


        # rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
        # print('Test RMSE: %.3f' % rmse)

        # return self.model.predict(X)
    
    def get_model(self):
        return self.model
    
    
        