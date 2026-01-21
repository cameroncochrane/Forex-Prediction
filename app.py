# Web app (frontend)
# Packages:
import streamlit as st
from data_functions import *
from arima_functions import *


# These dictionaries are needed to point the load model functions to the correct model paths
saved_arima_models = {'EURO/US$': 'models/EURO_US_arima_model.pkl',
 'UNITED KINGDOM POUND/US$': 'models/UNITED_KINGDOM_POUND_US_arima_model.pkl',
 'YEN/US$': 'models/YEN_US_arima_model.pkl',
 'YUAN/US$': 'models/YUAN_US_arima_model.pkl',
 'AUSTRALIAN DOLLAR/US$': 'models/AUSTRALIAN_DOLLAR_US_arima_model.pkl'}

saved_gb_models ={} ## UPDATE AS REQUIRED ##
saved_lstm_models ={} ## UPDATE AS REQUIRED

def master_():

    st.title("Forex Prediction App")

    # Define default currency and model so the app doesn't moan when booted!!!
    global currency
    global model_name #Model name, not the actual model (model_)
    # Define default currency
    currency = country_currency_dict['EURO AREA']
    model_name = "ARIMA"

    currency = st.selectbox(
        "Select a Currency",
        options=list(country_currency_dict.values()),
        on_change = load_model(currency,model_name)
        )
    
    model_name = st.selectbox(
        "Select a Model",
        options=["ARIMA","Gradient Boost","LSTM-RNN"],
        on_change=load_model(currency,model_name)
        )

    forecast_length = st.slider("Select forecast period (in days)", min_value=1, max_value=30, value=7)
    st.write(f"Forecast length: {forecast_length} days")


# Global function(s)

# This loads a model into the global model_ variable ready to be used for forecasting
def load_model(_currency,model):
    global model_ #To make sure the model object can be accessed anywhere...

    if model=="ARIMA":
        loaded_arima_models = load_saved_arima_models(saved_arima_models)
        model_name == "ARIMA"
        model_ = loaded_arima_models[_currency]
    if model =="Gradient Boost":
        print(1)
    if model =="LSTM-RNN":
        print(1)

def execute_forecast():
    print(1)
        

if __name__ == "__main__":
    # Run the Streamlit app
    global data
    global country_currency_dict
    global country_names

    data, country_currency_dict, country_names = load_and_process_forex_data()
    master_() #Executes when the app is opened.
    
    