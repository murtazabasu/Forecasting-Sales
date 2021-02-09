import streamlit as st
import functions
import pandas as pd
import config
import plotly.express as px


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

def write():
    st.title("Data Exploration")
    # Create a text element and let the reader know the data is loading.
    with st.spinner("Loading Data...."):
        # Load data into the dataframe.
        df_orig, _ = load_data()

    if st.checkbox('Show featured data (original)'):
        st.subheader('Raw Data')
        st.write(df_orig)

    # Sort data based on a date range
    st.subheader("Sort data based on a date")
    temp = df_orig.copy() 
    temp = temp.set_index('date')
    dates_ = list(temp.index.date)
    date_ = st.date_input("Please Enter range of Dates", [dates_[0], dates_[-1]])

    if (date_[0] and date_[1]) not in dates_:
        st.warning(
            f"Date data not available. Please select a date between {dates_[0]} and {dates_[0]}"
        )
    temp_date = temp[date_[0]:date_[1]].reset_index() 
    
    if st.checkbox("Show filtered data"):
        st.write(temp_date)

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

    st.markdown("<h2 style='text-align: center; color: Green;'>Correlations in Features", unsafe_allow_html=True)
    heat_map = functions.generate_correlation(df_orig, functions.columns_)
    st.pyplot(heat_map)

if __name__ == '__main__':
    write()