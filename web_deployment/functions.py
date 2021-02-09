import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# list of columns
columns_ = [
    'date',
    'seasonName', 
    'newSnowDescription',
    'rainDescription',
    'sunShineDescription',
    'temperatureDescription',
    'weekDayDescription',
]

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    :param data: dataframe
    :param n_in: number of input to be given to the model
    :param n_out = number of output from the model
    :param dropna: to drop the nan values
    :return: the aggregated rows 
    """
    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
	
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
	
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
	
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Function to compute mean absolute percentage error
def mean_absolute_percentage_error(y_true, y_pred): 
    """
    :param y_true: expected value
    :param y_pred: predicted value
    :return: MAPE
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

## columns uniquification
def columns_uniquification(data):
    """
    :param data: input data for unquifying columns
    :return data: dataset with unique columns 
    """
    cols = pd.Series(data.columns)
    for dup in data.columns[data.columns.duplicated(keep=False)]: 
        cols[data.columns.get_loc(dup)] = ([dup + '.' + str(d_idx) 
                                        if d_idx != 0 
                                        else dup 
                                        for d_idx in range(data.columns.get_loc(dup).sum())]
                                        )

    data.columns = cols
    return data

# generates correlation heatmap
def generate_correlation(data, columns):
    """
    :param data: dataset 
    :param return: correlation figure (heatmap)
    """
    fig, ax = plt.subplots()
    pearson_corr = data.drop(columns, axis=1).corr(method='pearson')
    ax = sns.heatmap(
        pearson_corr,
        vmax = 0.6,
        center = 0,
        square = True,
        linewidth = 0.5,
        cbar_kws = {"shrink":0.5},
        annot = True,
        fmt = '.2f',
        cmap = 'coolwarm' 
    )
    ax.figure.set_size_inches(12,12)

    return fig