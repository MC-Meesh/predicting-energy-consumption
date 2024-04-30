# Predicting Energy Consumption with LSTMs

## Problem Statement
Predicting energy consumption is vital for effective energy management, enabling energy-saving measures, and enhancing the efficiency of energy distribution systems. It supports the integration of renewable energy sources into the grid, aligning energy supply with demand, and reducing reliance on non-renewable, environmentally harmful sources. Accurate forecasts can lead to cost savings for both energy producers and consumers through optimized production and consumption strategies. Using data from the [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), Machine Learning models like LSTMs can be created to predict the energy consumption over the course of days, hours, or even minutes.

## Details
- Michael Chase Allen, CWID: 10857234
- The primary language used in this project is **Python**

## Code Structure
### Code

`main.py`: This is the entry point of the project. It parses command-line arguments, sets up the outputs directory, and orchestrates the data loading, exploratory data analysis (EDA), data preprocessing, model creation, fitting, prediction, and saving. It uses the other modules (data_loader, eda, data_preprocessor, and LSTM) to perform these tasks.

`data_loader.py`: Defines a *DataLoader* class responsible for loading, cleaning, and retrieving the data.

`eda.py`: This file contains code for performing exploratory data analysis (EDA) on the dataset but defining a *DataFramePlotter* class.

`data_preprocessor.py`: Contains a definition for the *TimeSeriesPreprocessor* class, responsible for cleaning the data and creating lag features used to feed into RNN-based architectures  like LSTMs.

`LSTM.py`: This file defines the *MyLSTM* class, which encapsulates the functionality of a Long Short-Term Memory (LSTM) model. The class includes methods for splitting the data, creating the model, fitting the model, making predictions, getting the model, and saving the model.'

### Other

- An **outputs/** folder is defined for generated figures and saving the LSTM model.
- A path to the data file **household_power_consumption.txt** needs to be specified when running main

## Running the Code

1. Ensure you have the [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) saved at an accessible path on your system (I reccomend in the root of this project in a direcotry named 'data').

2. Ensure you have all dependencies installed in your current Python environment by running `pip install -r requirements.txt`. If you still have depedency errors install the appropriate modules as needed.

3. Call the main script from the CLI, here is the general format:

```
usage: main.py [-h] [-show_data] [-eda] [-lag LAG] [-split SPLIT] [-save_model SAVE_MODEL] file_path

positional arguments:
    file_path             Path to the data file

optional arguments:
    -h, --help             show this help message and exit
    -show_data             Flag to enable displaying data in CLI and saving residual plots
    -eda                   Flag to enable EDA
    -lag LAG               Number of lag features
    -split SPLIT           Fraction for train/test split
    -save_model SAVE_MODEL Flag to enable saving the model to a specified folder name
```

An example would be: `python3 main.py data\household_power_consumption.txt -show_data -eda -lag 3` which would read the data from *data\household_power_consumption.txt*, perform some eda on the raw data, print data in the CLI and save detail plots to **outputs/**, and create 3 lag features for each input featue to feed into the LSTM.