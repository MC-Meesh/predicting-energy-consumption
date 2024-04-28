import data_loader as dl


import matplotlib.pyplot as plt

## Load and Clean Data

file_path = 'data/household_power_consumption.txt' #define path

data_loader = dl.DataLoader(file_path) #instantiate data loader
data_loader.load_data() #load data
data_loader.clean_data() #clean data
df = data_loader.get_data() #get data

## Data Analysis



## Define Model

## Train Model

## Evaluate Model