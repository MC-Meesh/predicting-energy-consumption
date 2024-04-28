import pandas as pd
import utils

class DataLoader:
    def __init__(self, file_path, show_data=False):
        self.file_path = file_path
        self.df = None
        self.show_data = show_data

    def load_data(self):
        utils.print_color('Loading data from:', self.file_path, color='green')
        self.df = pd.read_csv(self.file_path, sep=';', parse_dates={'dt':['Date','Time']},
                              infer_datetime_format=True, low_memory=False, na_values=['nan','?'],
                              index_col='dt')
        utils.print_color('Data loaded', color='green')
        
        if self.show_data:
            print(self.df.head())
        
        return self.df
    
    def clean_data(self):
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column].fillna(self.df[column].mode()[0], inplace=True) #impute missing values with most frequent value
                utils.print_bold_vars('Imputed missing values in column:', column, 'with mode:', self.df[column].mode()[0])
            else:
                self.df[column].fillna(self.df[column].mean(), inplace=True) #impute missing values with mean of column
                utils.print_bold_vars('Imputed missing values in column:', column, 'with mean:', self.df[column].mean())
        
        utils.print_color('NaN values imputed', color='green')

        if self.df.isna().sum().sum() != 0:
            utils.print_color(f'Warning: There are {self.df.isna().sum().sum()} NaN value in the data after imputing', color='yellow')

        utils.print_color('Data cleaned', color='green')

        if self.show_data:
            print(self.df.head())

        return self.df

    def get_data(self):
        return self.df
    
    def set_data(self, df):
        self.df = df
        utils.print_bold('Data set to input DataFrame')
        return self.df
    
    def get_data_shape(self):
        return self.df.shape
    
    def get_data_columns(self):
        return self.df.columns