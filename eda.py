import utils

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

class DataFramePlotter:
    def __init__(self, df, x_column):
        self.df = df
        self.x_column = x_column

    def plot(self):
        utils.print_color('Plotting data, this may take a moment...', color='green')

        n = len(self.df.columns) - 1
        ncols = 2
        nrows = math.ceil(n / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows))

        for ax, column in zip(axs.flatten(), self.df.columns):
            if column != self.x_column:
                if isinstance(self.df[column].dtype, pd.CategoricalDtype):
                    self.plot_categorical(ax, column)
                else:
                    self.plot_quantitative(ax, column)

        plt.tight_layout()
        plt.savefig('eda.png')
        
        utils.print_color('Plotting complete, saved to eda.png' , color='green')

    def plot_categorical(self, ax, column):
        sns.countplot(x=column, data=self.df, ax=ax)
        ax.set_title(f'Bar Chart of {column}')

    def plot_quantitative(self, ax, column):
        sns.histplot(self.df[column], kde=True, ax=ax)
        ax.set_title(f'Histogram of {column}')