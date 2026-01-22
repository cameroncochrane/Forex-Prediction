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

saved_gb_models ={} ## UPDATE AS REQUIRED, check identical structure to above ##
saved_lstm_models ={} ## UPDATE AS REQUIRED, " ##

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
        )
    
    model_name = st.selectbox(
        "Select a Model",
        options=["ARIMA","Gradient Boost","LSTM-RNN"],
        )
    
    st.button(
        label = "Load Model",
        on_click=load_model(currency,model_name)
    )

    global forecast_length
    forecast_length = st.slider("Select forecast period (in days)", min_value=1, max_value=30, value=1)
    st.write(f"Forecast length: {forecast_length} days")

    st.button(
        label="Forecast",
        on_click=execute_model
        )
    


# Global function(s)

# This loads a model into the global model_ variable ready to be used for forecasting
def load_model(_currency,model):
    global model_ #To make sure the model object can be accessed anywhere...

    if model=="ARIMA":
        loaded_arima_models = load_saved_arima_models(saved_arima_models)
        model_name == "ARIMA"
        model_ = loaded_arima_models[_currency]
        print(f"LOADED ARIMA MODEL:{_currency}")
    if model =="Gradient Boost":
        print(f"LOADED GB MODEL:{_currency}")
    if model =="LSTM-RNN":
        print(f"LOADED LSTMRNN MODEL:{_currency}")

def execute_model():
    """
    To be executed when forecast button is pressed
    """
    if model_name == "ARIMA":
        execute_arima(data)
    if model_name == "Gradient Boost":
        execute_xgboost(data)
        print()
    if model_name == "LSTM-RNN":
        print()
    else:
        print("Couldn't execute forecast")
    

def execute_arima(data):
    """
    To execute ARIMA forecast, we need model_(global), forecast_length(global), currency_(global), and access to data to generate
    a last_value.
    """
    # last_values = arima_generate_last_values()
    # forecasted_list = arima_single_model_forecast()
    # forecasted_df = arima_price_forecasts_to_dataframe()

    # global forecasted_data
    # forecasted_data = forecasted_df
    print()

def execute_xgboost(data):
    print()
        

if __name__ == "__main__":
    # Run the Streamlit app
    global data
    global country_currency_dict
    global country_names

    data, country_currency_dict, country_names = load_and_process_forex_data()
    master_() #Executes when the app is opened.
    
    