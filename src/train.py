import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.utils import check_array
import xgboost as xgb
import model_dispatcher
import config

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyoff
import plotly.graph_objs as go
import plotly.express as px
import joblib

matplotlib.rcParams['figure.figsize'] = (10,8)

columns = [
    'date',
    'seasonName', 
    'newSnowDescription',
    'rainDescription',
    'sunShineDescription',
    'temperatureDescription',
    'weekDayDescription',
]

# transform a time series dataset into a supervised learning dataset
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
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    """
    :param data: input dataframe
    :param data: input test split
    """
    return data[:-n_test, :], data[-n_test:, :]

def mean_absolute_percentage_error(y_true, y_pred): 
    """
    :param y_true: expected value
    :param y_pred: predicted value
    :return: MAPE
    """
    print(y_true)
    print(y_pred)
    #y_true, y_pred = check_array(np.asarray(y_true), np.asarray(y_pred))

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, model_name):
    """
    :param data: dataframe
    :param n_test: split amount for test data
    """
    predictions = list()

	# split dataset
    train, test = train_test_split(data, n_test)

    # seed history with training dataset
    history = [x for x in train]

    # step over each time-step in the test set
    for i in range(len(test)):
		# split test row into input and output columns
        testX, testy = test[i, 1:], test[i, 0]
        
        # fit model on history and make a prediction
        train_ = np.asarray(history)
        trainX, trainy = train_[:, 1:], train_[:, 0]
        model = model_dispatcher.models[model_name]
        model.fit(trainX, trainy)

        # make a one-step prediction
        yhat = model.predict([testX])
        
		    # store forecast in list of predictions
        predictions.append(yhat[0])

		    # add actual observation to history for the next loop
        history.append(test[i])
		    
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat[0]))
	
    # save the model
    #joblib.dump(model, config.MODEL_OUTPUT + model_name + '_2_' + ".joblib")
    # estimate prediction error
    error = mean_absolute_percentage_error(test[:, 0], predictions)
    return error, test[:, 0], predictions


if __name__ == '__main__':

    # load the dataset
    df_ = pd.read_csv(config.INPUT_DATA, delimiter=';', parse_dates=['date'])
    df_ = df_.sort_values(by="date")
    df_ = df_.drop(columns, axis=1)

    values = df_.values

    model_list = ["random_forest"]

    for model in model_list:
        # transform the time series data into supervised learning
        data = series_to_supervised(values, n_in=6)

        # evaluate
        mae, y, yhat = walk_forward_validation(data, 2, model)
        print('MAE: %.3f' % mae)

        # plot expected vs predicted
        plt.plot(y, label='Expected')
        plt.plot(yhat, label='Predicted')
        plt.legend()
        plt.savefig(config.OUTPUT + model)
        plt.close('all')
