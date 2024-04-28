import data_loader as dl

import matplotlib.pyplot as plt

## Load and Clean Data

file_path = 'data/household_power_consumption.txt' #define path

data_loader = dl.DataLoader(file_path) #instantiate data loader
data_loader.load_data() #load data
df = data_loader.get_data() #get data
data_loader.clean_data() #clean data

print(df.isna().sum())

## Data Analysis

def plot_column():
    # Plot a column of the time series data
    plt.plot(df['Global_active_power'])
    plt.xlabel('Time')
    plt.ylabel('Column Value')
    plt.title('Time Series Data')
    plt.show()

plot_column()

## Define Model

## Train Model

## Evaluate Model