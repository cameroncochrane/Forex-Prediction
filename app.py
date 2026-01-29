# Web app (frontend)
# Packages:
import streamlit as st
import pandas as pd
from data_functions import *
from arima_functions import *
from xgboost_functions import *


# These dictionaries are needed to point the load model functions to the correct model paths
saved_arima_models = {'EURO/US$': 'models/EURO_US_arima_model.pkl',
 'UNITED KINGDOM POUND/US$': 'models/UNITED_KINGDOM_POUND_US_arima_model.pkl',
 'YEN/US$': 'models/YEN_US_arima_model.pkl',
 'YUAN/US$': 'models/YUAN_US_arima_model.pkl',
 'AUSTRALIAN DOLLAR/US$': 'models/AUSTRALIAN_DOLLAR_US_arima_model.pkl'}

saved_gb_models ={'EURO/US$': 'models/EURO_US_xgboost_model.pkl',
 'UNITED KINGDOM POUND/US$': 'models/UNITED_KINGDOM_POUND_US_xgboost_model.pkl',
 'YEN/US$': 'models/YEN_US_xgboost_model.pkl',
 'YUAN/US$': 'models/YUAN_US_xgboost_model.pkl',
 'AUSTRALIAN DOLLAR/US$': 'models/AUSTRALIAN_DOLLAR_US_xgboost_model.pkl'}


saved_lstm_models ={} ## UPDATE AS REQUIRED, " ##
# (session state will hold the most recent forecast dataframe)

def master_():

    st.title("Forex Prediction App")

    # Define default currency
    global currency
    currency = country_currency_dict['EURO AREA']
    global model_name
    model_name = "ARIMA"

    # Allow the user to select currency and model:
    currency = st.selectbox(
        "Select a Currency",
        options=list(country_currency_dict.values()),
        )
    
    model_name = st.selectbox(
        "Select a Model",
        options=["ARIMA","Gradient Boost","LSTM-RNN"],
        )
    
    # Use on_click with a callable and args to avoid calling load_model at render time
    st.button(label="Load Model", on_click=load_model, args=(currency, model_name)) # load_model will be called only when button is clicked

    global forecast_length
    forecast_length = st.slider("Select forecast period (in days)", min_value=1, max_value=30, value=1)
    st.write(f"Forecast length: {forecast_length} days")

    st.button(
        label="Forecast",
        on_click=execute_model
        )

    # Ensure a persistent slot for forecasted data across reruns
    if 'forecasted_data' not in st.session_state:
        st.session_state['forecasted_data'] = None

    # Display forecasted dataframe if available in session state
    forecasted = st.session_state.get('forecasted_data')
    if forecasted is not None:
        st.subheader("Forecasted Data")
        st.dataframe(forecasted)
        
        # Always attempt to plot historical data (excluding the last 60 rows)
        # alongside the forecast on a numeric index axis.
        try:
            test_len = 60
            train_df = data.iloc[:-test_len].copy()

            # Historical series with Date index — only keep recent rows so forecasts are visible
            hist = train_df[['Date', currency]].set_index('Date')
            max_hist = 200
            hist = hist.tail(max_hist)

            # Forecast dataframe: keep its integer index (1..N) as produced by ARIMA helper
            fc = forecasted.copy()

            # Rename columns for clarity: historical vs forecast
            hist = hist.rename(columns={currency: 'Historical'})
            first_col = fc.columns[0]
            fc = fc.rename(columns={first_col: 'Forecast'})

            # Reindex historical data to integer indices so last historical point is 0
            m = len(hist)
            hist.index = range(-m + 1, 1)  # e.g. -{m-1} ... -1, 0

            # Forecast already has index 1..n, so concat will produce a numeric index
            combined = pd.concat([hist, fc[['Forecast']]], axis=1)
            st.line_chart(combined)
        except Exception as e:
            st.write("Could not plot combined chart:", e)
    


# Global function(s)

# This loads a model into the global model_ variable ready to be used for forecasting
def load_model(_currency,model_n):
    global model_name

    # persist model in session_state so it survives reruns
    if 'model_' not in st.session_state:
        st.session_state['model_'] = None

    if model_n=="ARIMA":
        loaded_arima_models = load_saved_arima_models(saved_arima_models)
        model_name = "ARIMA"
        st.session_state['model_'] = loaded_arima_models[_currency]
        print(f"LOADED ARIMA MODEL:{_currency}")
    if model_n =="Gradient Boost":
        loaded_gb_models = load_saved_gb_models(saved_gb_models)
        model_name = "Gradient Boost"
        st.session_state['model_'] = loaded_gb_models[_currency]
        print(f"LOADED GB MODEL:{_currency}")
    if model_n =="LSTM-RNN":
        print(f"LOADED LSTMRNN MODEL:{_currency}")
        print("Cannot load (model not saved yet)") # Remove when models are ready for use

def execute_model():
    """To be executed when forecast button is pressed"""
    if model_name == "ARIMA":
        model_obj = st.session_state.get('model_')
        if model_obj is not None:
            fd = execute_arima(model=model_obj, data=data, currency=currency, f=forecast_length)
            # persist forecast in session state so it survives reruns
            st.session_state['forecasted_data'] = fd
        else:
            print(model_obj)
            print("Load a model")

    elif model_name == "Gradient Boost":
        model_obj = st.session_state.get('model_')
        if model_obj is not None:
            fd = execute_xgboost(model=model_obj, xgbdata_p=xgbdata_processed, currency=currency, f=forecast_length)
            # persist forecast in session state so it survives reruns
            st.session_state['forecasted_data'] = fd
        else:
            print(model_obj)
            print("Load a model")

    elif model_name == "LSTM-RNN":
        # Make sure to add a way to load the scaler used during training here ready for forecast value inversion.
        pass
    else:
        print("Couldn't execute forecast")
    
def execute_arima(model,data,currency,f):
    """
    To execute ARIMA forecast, we need model_(global), forecast_length(global), currency(global), and access to data to generate
    a last_value.
    """
    all_last_values = arima_generate_last_values(data=data,country_currency_dict=country_currency_dict,test_length=60)
    last_values_single = all_last_values[currency]
    single_forecasted_list = arima_single_model_forecast(model=model,last_value=last_values_single,forecast_length=f)
    
    fd = arima_price_forecast_to_dataframe(single_forecasted_list)
    print(fd.shape)
    return fd
    
def execute_xgboost(model,xgbdata_p,currency,f):

    bundle = xgbdata_p[currency] #Generated via: processed_xgb_data = process_all_xgboost(raw_gradient_data)

    fd = make_recursive_forecast_xgboost(
        xgb_model=model,
        df_features=bundle["df_features"],
        currency_col=bundle["df_features"].columns[0], # or the known currency column name
        feature_cols=bundle["feature_cols"],
        forecast_length=f
        )

    return fd


# PLACE RNN FORECASTING EXECUTE + SCALER INVERSION FOR DF OUTPUT HERE:
def execute_rnn(model,rnndata_p,currency,f,scaler):

    return None

##########################################################################

if __name__ == "__main__":
    # Run the Streamlit app
    global data
    global country_currency_dict
    global country_names
    global xgbdata_processed

    # All data:
    data, country_currency_dict, country_names = load_and_process_forex_data()

    # XGBoost Specific data organisation:
    xgbdata_raw = create_data_dict_currency_xgboost(data,country_names,country_currency_dict)
    xgbdata_processed = process_all_xgboost(xgbdata_raw)

    # RNN specific data organisation:

    master_() #Executes when the app is opened.
    
    