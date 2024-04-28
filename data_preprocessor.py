import pandas as pd

class TimeSeriesPreprocessor:
    def __init__(self, data):
        self.df = pd.DataFrame(data)

    # Convert series to supervised learning, method from Air_Pollution_LSTM project by 'Juned-the-programmer' - https://github.com/Juned-the-programmer/Air_Pollution_LSTM/blob/fd3abbbc43dae58388b7a947d2b1f5b493611997/LSTM%20Forecasting/test.py#L41-L46
    def series_to_supervised(self, n_in=1, n_out=1, dropnan=True):
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
        return agg