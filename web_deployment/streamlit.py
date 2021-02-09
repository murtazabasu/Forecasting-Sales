import streamlit as st
import awesome_streamlit as ast
import pandas as pd
import numpy as np
import plotly.express as px
import config
import pickle
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

ast.core.services.other.set_logging_format()

#st.set_page_config(layout="centered")

image_logo = Image.open(config.MEDIA + 'logo_2.jpg')
image_tomatoes = Image.open(config.MEDIA + 'tomatoes.jpg')

st.image(
    image_logo,
    use_column_width=True
)

st.write('\n')
st.markdown("<h1 style='text-align: center; color: Green;'>Case Study: Forecasting Daily Sales of Tomatoes", unsafe_allow_html=True)

st.write('\n')
st.markdown("<h4 style='text-align: justify; color: Green;'>A customer wants to forecast the sales of tomatoes on a daily basis (1-day-ahead), aggregated over approximately 100\
stores in order to optimize supply chain planning. Currently, the customer uses a simple forecast, he uses the sales one week before on the same weekday as forecast for the\
week to come (for e.g. forecast for coming Tuesday = sales last Tuesday).", unsafe_allow_html=True)
st.write('\n')
st.markdown("<h4 style='text-align: justify; color: Green;'>We have developed a model using machine learning that takes different features from the data as\
input and gives a more better forecast for the sales of the tomatoes for the next day.", unsafe_allow_html=True)
st.write('\n')
st.image(
    image_tomatoes,
    use_column_width=True
)

# model file
model_file = 'rforest_500_6.pickle'

####################################################################
####################################################################
####################################################################
####################################################################

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

def generate_correlation(data, columns):
    """
    :param data: dataset 
    :param return: figure
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

####################################################################
####################################################################
####################################################################
####################################################################

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

fig = plt.figure()

df_orig = pd.read_csv(config.INPUT_DATA + 'tomatoesAndFeatures.csv', delimiter=';', parse_dates=['date'])
df_orig = df_orig.sort_values(by="date")
df_orig = df_orig.drop(columns_, axis=1)

# extract only the values from the original df
values_orig = df_orig.values

# load customer data
df_customer = pd.read_csv(config.INPUT_DATA + 'tomatoes_CustomerData.csv', delimiter=',', parse_dates=['date'])
df_customer = df_customer.sort_values(by="date")
df_customer = df_customer.drop(['date'], axis=1)

@st.cache
def load_data():
    # load the original dataset
    df_orig = pd.read_csv(config.INPUT_DATA + 'tomatoesAndFeatures.csv', delimiter=';', parse_dates=['date'])
    df_orig = df_orig.sort_values(by="date")
    #df_orig = df_orig.drop(columns, axis=1)

    # load customer data
    df_customer = pd.read_csv(config.INPUT_DATA + 'tomatoes_CustomerData.csv', delimiter=',', parse_dates=['date'])
    df_customer = df_customer.sort_values(by="date")
    #df_customer = df_customer.drop(['date'], axis=1)

    return df_orig, df_customer

# Create a text element and let the reader know the data is loading.
with st.spinner("Loading Data...."):
    data_load_state = st.text('Daten laden...')

# Load 10,000 rows of data into the dataframe.
df_orig, df_customer = load_data()

if st.checkbox('Show featured data (original)'):
    st.subheader('Raw Data')
    st.write(df_orig)

heat_map = generate_correlation(df_orig, columns_)
st.pyplot(heat_map)

# Sort data based on a date range
if st.checkbox('Sort data based on a date'):
    temp = df_orig.copy() 
    temp = temp.set_index('date')
    dates_ = list(temp.index.date)
    date_ = st.date_input("Please Enter range of Dates", [dates_[0], dates_[-1]])

    if (date_[0] and date_[1]) not in dates_:
        st.warning(
            f"Date data not available. Please select a date between {dates_[0]} and {dates_[0]}"
        )
    temp_date = temp[date_[0]:date_[1]].reset_index() 
    st.write(temp_date)

    if st.checkbox('Show sales amount for this duration'):
        st.subheader(f'Sales amount from {date_[0]} to {date_[1]}')
        fig = px.line(
            temp_date, 
            x='date', 
            y='salesAmount', 
            title='Amount of Sales',
            width=800,
            height=600
        )
        fig.data[0].update(mode='lines+markers')
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig)


st.write('\n')
# OUR PREDICTIONS
st.markdown("<h3 style='text-align: center; color: Green;'>-------------------------------Our Forecast Model-------------------------------", unsafe_allow_html=True)
st.write('\n')

st.subheader('Select a date duration to display customer predictions')
n_steps = 6

# get the feature generated data and the customer data
featured_data = series_to_supervised(df_orig, n_in=n_steps) 
customer_data = df_customer.copy() 

# set index as date for the customer data
customer_data = customer_data.set_index('date')

# get list of dates
date_list = list(featured_data.iloc[:,0].to_list()) 

try:
    date_ = st.date_input("Please Enter range of Dates:", [date_list[0]])

    if (date_[0] and date_[1]) not in date_list:
        st.warning(
            f"Date data not available. Please select a date between {date_list[0].date()} and {date_list[-1].date()}"
        )

    # get the indices for the start and end date for the featured data
    start_indice = featured_data[featured_data.iloc[:,0]==str(date_[0])].index[0]
    end_indice = featured_data[featured_data.iloc[:,0]==str(date_[1])].index[0]

    # get the list of dates
    df_dates = featured_data.copy()
    dates_data = columns_uniquification(df_dates) 
    dates_list = dates_data.set_index('date')[str(date_[0]):str(date_[1])].index.to_list() 

    # slice the customer data based on the date range
    customer_data_ = customer_data[date_[0]:date_[1]].reset_index() 

    # transformed data
    sliced_data = featured_data.loc[start_indice-1:end_indice-1]

    # drop the date column with other redundant columns 
    sliced_data = sliced_data.drop(columns=columns_, axis=1)

    # get the features and the true target values (salesamount) for the featured data
    features, true_values = sliced_data.values[:,1:], sliced_data.values[:,0]

    # get the true sales amount and the customer predictions 
    # from the customer data
    c_true, c_preds = customer_data_['salesAmount'].values, customer_data_['sales_before_1_week'].values 

    # load the model
    with st.spinner('Loading model and accumulating predictions...'):
        model = pickle.load(open(config.MODEL_OUTPUT + model_file, 'rb')) 
        # get predictions form the model
        preds_ = []
        for f in features:
            pred = model.predict([f])
            preds_.append(pred[0])
        
    results_dataframe = pd.DataFrame(
        {
            'Date': dates_list,
            'Customer_Prediction': c_preds,
            'Model_Prediction': preds_,
            'Expected_Sale': true_values
        }
    )
    fig = px.scatter(
        results_dataframe, 
        x = 'Date',
        y=['Customer_Prediction', 'Model_Prediction', 'Expected_Sale'],
        title='Sales Forecasting Comparison',
        width=800,
        height=600
    )
    
    fig.data[0].update(mode='lines+markers')
    fig.data[1].update(mode='lines+markers')
    fig.data[2].update(mode='lines+markers')
    #fig.data[3].update(mode='lines+markers')
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)

    # Mean absolute percentage error for the model 
    mape_customer = mean_absolute_percentage_error(
        c_true,
        c_preds
    )

    # Mean absolute percentage error for the model  
    mape_model = mean_absolute_percentage_error(
        true_values,
        preds_
    )

    st.markdown(f"<h3 style='text-align: center; color: Green;'> MAPE in Customer Prediction: {mape_customer:.2f}%", unsafe_allow_html=True)
    st.write('\n')
    st.markdown(f"<h3 style='text-align: center; color: Green;'> MAPE of Our Model: {mape_model:.2f}%", unsafe_allow_html=True)
    st.write('\n')

except IndexError:
    st.warning("Date Out of Range!")

