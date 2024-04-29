'''
Michael (Chase) Allen
5/1/2024

main.py
'''

# User defined modules
import utils
import data_loader as dl
import data_preprocessor as dp
import eda as eda
import LSTM as lstm

# External modules
import sys
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('file_path', type=str, help='Path to the data file')
    parser.add_argument('-show_data', action='store_true', help='Flag to enable displaying data')
    parser.add_argument('-eda', action='store_true', help='Flag to enable EDA')
    parser.add_argument('-lag', type=int, default=1, help='Number of lag features')
    parser.add_argument('-split', type=float, default=.8, help='Fraction for train/test split')
    

    args = parser.parse_args()

    file_path = args.file_path
    doEDA = args.eda
    showData = args.show_data
    lag_count = args.lag
    split = args.split

    ## Load Data

    data_loader = dl.DataLoader(file_path, showData) #instantiate data loader
    data_loader.load_data() #load data
    data_loader.clean_data() #clean data
    df = data_loader.get_data() #get data

    ## Data Analysis

    if doEDA:
        plotter = eda.DataFramePlotter(df, 'dt') #instantiate DataFramePlotter
        plotter.plot() #plot data
    else:
        utils.print_color('EDA disabled, use "-eda" flag to enable', color='yellow')

    ## Preprocess Data

    preprocessor = dp.TimeSeriesPreprocessor(df, showData) #instantiate TimeSeriesPreprocessor
    preprocessor.resample_data('h') #resample data
    preprocessor.rescale_data() #could add feature range as a parameter
    preprocessor.series_to_supervised(n_in=lag_count) #preprocess data
    preprocessor.drop_columns() #drop (non-train || non-target) columns
    df = preprocessor.get_data() #get preprocessed data

    ## Define Model

    lstm_model = lstm.MyLSTM(df, showData, split=split) #instantiate LSTM class
    lstm_model.fit_model() #fit model
    lstm_model.predict(scaler=preprocessor.get_scaler()) #plot loss

    ## Train Model



    ## Evaluate Model