import utils
import numpy as np

from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
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
        model.add(LSTM(300, activation='relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def fit_model(self, epochs=20, batch_size=70):
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test), verbose=2, shuffle=False)
        self.history = history

        if self.show_data:
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            plt.savefig('outputs/model_loss.png')
            utils.print_color('Model loss saved to: model_loss.png', color='green')
        
        return history

    def predict(self, scaler):

        '''
        Method resembles the following snippet from Kaggle:
        https://www.kaggle.com/code/cuge1995/lstm-for-household-electric-power-cb225cfe-1?scriptVersionId=14243064&cellId=13
        '''
        # Get the number of features
        n_features = self.X_test.shape[2]

        # Make a prediction
        yhat = self.model.predict(self.X_test)

        # Reshape X_test
        self.X_test = self.X_test.reshape((self.X_test.shape[0], n_features))

        # Prepare data for inverse scaling
        yhat_data = np.concatenate((yhat, self.X_test[:, -n_features:]), axis=1)
        y_test_data = np.concatenate((self.y_test.reshape((len(self.y_test), 1)), self.X_test[:, -n_features:]), axis=1)

        # Invert scaling for forecast
        inv_yhat = scaler.inverse_transform(yhat_data)
        inv_yhat = inv_yhat[:,0]

        # Invert scaling for actual
        inv_y = scaler.inverse_transform(y_test_data)
        inv_y = inv_y[:,0]

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
        utils.print_color('Test RMSE: %.3f' % rmse, color='green')

        utils.print_color('Creating predictions for test data', color='green')
        model_preds_path = 'outputs/model_preds'
        # Prepare data for plot
        times = range(len(self.y_test))

        plt.figure(figsize=(18, 6)) # Need a larger figure size for the plot
        plt.plot(times, inv_y, marker='.', color=(38/255, 170/255, 231/255), label="actual")
        plt.plot(times, inv_yhat, marker='.', color=(231/255, 146/255, 38/255), label="prediction")
        plt.ylabel('Global_active_power', size=15)
        plt.xlabel('Time step', size=15)
        plt.legend(fontsize=15)
        plt.savefig(f'{model_preds_path}.png')
        utils.print_color(f'Predictions plot saved to: {model_preds_path}.png', color='green')

        if self.show_data:
            utils.print_color('Saving model evaluation plots [show_data enabled]', color='green')
            details_path = 'outputs/residual_details'

            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
            # Residual Plot
            residuals = inv_y - inv_yhat
            axs[0, 0].scatter(range(len(residuals)), residuals)
            axs[0, 0].set_xlabel('Time step')
            axs[0, 0].set_ylabel('Residual')
            axs[0, 0].set_title('Residual Plot')

            # Histogram of Residuals
            axs[0, 1].hist(residuals, bins=30)
            axs[0, 1].set_xlabel('Residual')
            axs[0, 1].set_ylabel('Frequency')
            axs[0, 1].set_title('Histogram of Residuals')

            # Actual vs Predicted
            axs[1, 0].scatter(inv_y, inv_yhat)
            axs[1, 0].set_xlabel('Actual')
            axs[1, 0].set_ylabel('Predicted')
            axs[1, 0].set_title('Actual vs Predicted')
            axs[1, 0].plot([min(inv_y), max(inv_y)], [min(inv_y), max(inv_y)], color='red')  # Diagonal line

            # RMSE Over Time
            axs[1, 1].plot(self.history.history['loss'])
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].set_title('MSE Over Time')

            plt.tight_layout()
            plt.savefig(f'{details_path}.png')
            utils.print_color(f'Details saved to: {details_path}.png', color='green')

    def get_model(self):
        return self.model
    
    def save_model(self, path):
        path = 'outputs/' + path
        if not path.endswith(('.keras', '.h5')):
            path += '.keras'
        self.model.save(path)
        utils.print_bold(f'Model saved to: {path}')

        