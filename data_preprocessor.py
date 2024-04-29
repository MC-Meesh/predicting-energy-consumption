import utils 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class TimeSeriesPreprocessor:
    def __init__(self, data, show_data=False):
        self.df = pd.DataFrame(data)
        self.show_data = show_data
        self.scaler = None

    def resample_data(self, freq):
        old_shape = self.df.shape # Get the shape of the dataframe before resampling

        freq_mapping = {
            'h': 'hour',
            'D': 'day',
            'W': 'week',
            'M': 'month',
            'Q': 'quarter',
            'Y': 'year'
        }
        freq_name = freq_mapping.get(freq, freq)
        utils.print_color(f'Resampling data to: {freq_name} frequency, taking mean', color='green')
        
        try:
            self.df = self.df.resample(freq).mean()
        except Exception as e:
            utils.print_color('Error resampling data:', color='red')
            print(e)
            return None
        
        utils.print_color('Data resampled', color='green')
        
        if self.show_data:
            utils.print_color(f'Old df shape: {old_shape}', color='yellow')
            utils.print_color(f'New df shape: {self.df.shape}', color='yellow')
            print(self.df.head())
        
        return self.df

    def rescale_data(self):
        utils.print_color('Rescaling data to feature range (0,1)', color='green') #could add feature range as a parameter
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(self.df)
        self.df = pd.DataFrame(scaled)
        utils.print_color('Data rescaled', color='green')
        if self.show_data:
            print(self.df.head())
        return self.df
    
    # Convert series to supervised learning, method from Air_Pollution_LSTM project by 'Juned-the-programmer' - https://github.com/Juned-the-programmer/Air_Pollution_LSTM/blob/fd3abbbc43dae58388b7a947d2b1f5b493611997/LSTM%20Forecasting/test.py#L41-L46
    def series_to_supervised(self, n_in=1, n_out=1, dropnan=True):
        utils.print_color(f'Converting series for supervised learning, using {n_in} previous time step(s)', color='green')
        n_vars = 1 if type(self.df) is list else self.df.shape[1]
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(self.df.shift(-i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(self.df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        self.df = agg
        utils.print_color('Series converted for supervised learning', color='green')
        if self.show_data:
            print(self.df.head())
        return agg
    
    def drop_columns(self):

        '''
        This one is interesting, we are dropping non lag columns [columns that are not varX(t-y) format] except for the target feature [var1(t)].
        '''

        # Identify lag and target columns
        lag_columns = [col for col in self.df.columns if '(t-' in col]
        target_column = 'var1(t)'

        # Identify columns to drop
        columns_to_drop = [col for col in self.df.columns if col not in lag_columns and col != target_column]

        utils.print_color(f'Dropping columns: {columns_to_drop}', color='green')
        self.df.drop(columns_to_drop, axis=1, inplace=True)
        utils.print_color('Columns dropped', color='green')
        

        if self.show_data:
            utils.print_bold_vars(f'New Shape: {self.df.shape}')
            print(self.df.head())
        
        return self.df
    
    def get_data(self):
        return self.df
        
    def get_scaler(self):
        return self.scaler