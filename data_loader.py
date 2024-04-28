import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path, sep=';', parse_dates={'dt':['Date','Time']},
                              infer_datetime_format=True, low_memory=False, na_values=['nan','?'],
                              index_col='dt')
        return self.df
    
    def clean_data(self):
        self.df = self.df.fillna(self.df.mean())
        self.df.dropna(inplace=True)
        return self.df

    def get_data(self):
        return self.df
    
    def get_data_shape(self):
        return self.df.shape
    
    def get_data_columns(self):
        return self.df.columns