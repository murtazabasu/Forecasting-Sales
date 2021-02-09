import streamlit as st
import functions
import pandas as pd
import config
import plotly.express as px
import pickle

n_steps = 6

# load the original dataset
df_orig = pd.read_csv(config.INPUT_DATA + 'tomatoesAndFeatures.csv', delimiter=';', parse_dates=['date'])
df_orig = df_orig.sort_values(by="date")

# load customer data
df_customer = pd.read_csv(config.INPUT_DATA + 'tomatoes_CustomerData.csv', delimiter=',', parse_dates=['date'])
df_customer = df_customer.sort_values(by="date")

def write():
    st.write("\n")
    st.title("Sales Forecasting Comparison between our Model and the Customer")
    
    # get the feature generated data and the customer data
    featured_data = functions.series_to_supervised(df_orig, n_in=n_steps) 
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
        dates_data = functions.columns_uniquification(df_dates) 
        dates_list = dates_data.set_index('date')[str(date_[0]):str(date_[1])].index.to_list() 

        # slice the customer data based on the date range
        customer_data_ = customer_data[date_[0]:date_[1]].reset_index() 

        # transformed data
        sliced_data = featured_data.loc[start_indice-1:end_indice-1]

        # drop the date column with other redundant columns 
        sliced_data = sliced_data.drop(columns=functions.columns_, axis=1)

        # get the features and the true target values (salesamount) for the featured data
        features, true_values = sliced_data.values[:,1:], sliced_data.values[:,0]

        # get the true sales amount and the customer predictions 
        # from the customer data
        c_true, c_preds = customer_data_['salesAmount'].values, customer_data_['sales_before_1_week'].values 

        # load the model
        with st.spinner('Loading model and accumulating predictions...'):
            model = pickle.load(open(config.MODEL_OUTPUT + config.model_file, 'rb')) 
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
        mape_customer = functions.mean_absolute_percentage_error(
            c_true,
            c_preds
        )

        # Mean absolute percentage error for the model  
        mape_model = functions.mean_absolute_percentage_error(
            true_values,
            preds_
        )

        st.markdown(f"<h2 style='text-align: center; color: Green;'> MAPE (Mean Absolute Percentage Error)", unsafe_allow_html=True)
        st.write('\n')
        st.markdown(f"<h3 style='text-align: center; color: Black;'> Model Prediction: {mape_model:.2f}% | Customer Prediction: {mape_customer:.2f}%", unsafe_allow_html=True)
        st.write('\n')

    except IndexError:
        st.warning("Date Out of Range!")


if __name__ == '__main__':
    write()