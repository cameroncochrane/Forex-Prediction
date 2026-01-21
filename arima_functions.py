import pandas as pd
import numpy as np
from data_functions import *
import pickle
import os

def prep_for_arima_log_returns(prices: pd.Series) -> pd.Series:
    
    s = pd.to_numeric(prices.sort_index(), errors="coerce").dropna()
    y = np.log(s).diff().dropna()
    y.name = "y"
    return y

def log_returns_to_price_path_arima(
    predicted_log_returns: pd.Series,
    last_close: float,
    start_price_name: str = "price",
):
    """
    Convert predicted log returns into a price forecast path anchored at last_close.

    Parameters
    ----------
    predicted_log_returns : pd.Series
        Predicted log returns for future steps (e.g., from ARIMA forecast).
        Index can be dates; if not, a RangeIndex is fine.
    last_close : float
        Last observed close price (anchor).
        
    Returns
    -------
    pd.Series
        Forecasted prices (same index as predicted_log_returns).
    """
  
    r = predicted_log_returns.astype(float).copy()

    # log price path: log(P_t+k) = log(P_t) + sum_{i=1..k} r_{t+i}
    log_price = np.log(float(last_close)) + r.cumsum()
    prices = np.exp(log_price)

    prices.name = start_price_name
    return prices

def train_arima(training_dictionary,p=10,q=10):

    currencies = list(training_dictionary.keys())

    ARIMA_models = {}

    for currency in currencies:
        log_d = training_dictionary[currency]
        ARIMA_currency_model = ARIMA(log_d, order=(p, 0, q)).fit()
        ARIMA_models[currency] = ARIMA_currency_model

    return ARIMA_models

def save_trained_models_arima(trained_models_dict, save_directory="models"):
    """
    Save trained ARIMA models to pickle files.
    
    Parameters:
    -----------
    trained_models_dict : dict
        Dictionary containing trained models with currency names as keys
    save_directory : str, default="models"
        Directory to save the model files
    
    Returns:
    --------
    dict: Dictionary mapping currency names to their saved file paths
    """
    
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    saved_files = {}
    
    for currency_name, model in trained_models_dict.items():
        # Clean currency name for filename (replace special characters)
        clean_name = currency_name.replace("/", "_").replace("$", "").replace(" ", "_")
        filename = f"{clean_name}_arima_model.pkl"
        filepath = os.path.join(save_directory, filename)
        
        # Save model using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        saved_files[currency_name] = filepath
        print(f"Saved {currency_name} model to {filepath}")
    
    return saved_files

def load_saved_arima_models(saved_files_dict):
    """
    Load saved ARIMA models from pickle files.
    
    Parameters:
    -----------
    saved_files_dict : dict
        Dictionary mapping currency names to their saved file paths
        (output from save_trained_models function)
    
    Returns:
    --------
    dict: Dictionary containing loaded models with currency names as keys
    """
    
    loaded_models = {}
    
    for currency_name, filepath in saved_files_dict.items():
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            loaded_models[currency_name] = model
            print(f"Successfully loaded {currency_name} model from {filepath}")
        except FileNotFoundError:
            print(f"Error: Could not find file {filepath} for {currency_name}")
        except Exception as e:
            print(f"Error loading {currency_name} model: {str(e)}")
    
    return loaded_models

def arima_generate_last_values(train_data,country_currency_dict):
    """
    Find the last values in the training data to allow conversion of forecasted log returns to forecasted prices

    country_currency_dict: Generated when importing the data.
    
    """
    c_list = list(country_currency_dict.values())
    training_data_last_values = {}

    for currency in c_list:
        # Get the last value from training data
        last_value = train_data[currency].iloc[-1]
        training_data_last_values[currency] = last_value
    
    return training_data_last_values

def arima_models_forecast(model_dict,last_values,forecast_length=60):

    currencies = list(model_dict.keys())
    
    currency_log_forecasts = {}
    currency_price_forecasts = {}

    for currency in currencies:
        m = model_dict[currency]
        currency_log_forecasts[currency] = m.forecast(forecast_length)
        last_price = last_values[currency]
        currency_price_forecasts[currency] = log_returns_to_price_path_arima(currency_log_forecasts[currency], last_price, f"{currency}_predicted_price")
    
    return currency_price_forecasts

def arima_single_model_forecast(model, last_value, forecast_length=60, currency_name="currency"):
    """
    Forecast future prices using a single ARIMA model.

    Parameters
    ----------
    model : ARIMAResults
        Trained ARIMA model for the currency.
    last_value : float
        Last observed price for the currency.
    forecast_length : int, optional
        Number of steps to forecast, by default 60.
    currency_name : str, optional
        Name of the currency, used for naming the forecasted price series, by default "currency".

    Returns
    -------
    pd.Series
        Forecasted prices for the given currency.
    """
    # Forecast log returns
    currency_log_forecast = model.forecast(forecast_length)
    
    # Convert log returns to price path
    currency_price_forecast = log_returns_to_price_path_arima(
        currency_log_forecast, last_value, f"{currency_name}_predicted_price"
    )
    
    return currency_price_forecast

def arima_price_forecasts_to_dataframe(currency_price_forecasts_dict):
    # Create DataFrame from the dictionary
    df = pd.DataFrame(currency_price_forecasts_dict)

    # Reset index to create a forecast period column
    df = df.reset_index()
    df = df.rename(columns={'index': 'forecast_period'})

    # Set forecast period as sequential numbers starting from 1
    df['forecast_period'] = range(1, len(df) + 1)
    df = df.set_index('forecast_period')

    return df
