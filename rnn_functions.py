import pandas as pd
import numpy as np

def extract_currency_df_rnn(data, currency):
    """
    Extract a specific country's data from the main (unprocessed) dataframe
    (This function is called in the 'create_data_dict_currency_rnn()' function).
    
    Parameters:
    data (DataFrame): The main dataframe containing all countries' data
    country_name (str): The name of the country to extract data for
    
    Returns:
    DataFrame: A dataframe containing only the specified country's data with date column
    """
    country_data = data[['Date',currency]].copy()
    
    # Ensure date column is included and properly formatted
    
    return country_data


def create_data_dict_currency_rnn(data,countries,currency_dict):
    """
    Uses extract_currency_df_rnn() to create individual dataframes for each currency and places
    them in a dictionary of dataframes, ready for pre-processing for XGBoost modelling.

    Example:
    rnndata_raw = create_data_dict_currency_rnn(data,country_names,country_currency_dict)

    Where data, country_names, country_currency_dict are created when the app.py script is run from the 'master' data import call.
    """

    data_dict = {}
    for country in countries:
        data_dict[currency_dict[country]] = extract_currency_df_rnn(data,currency_dict[country])
    
    return data_dict


def process_all_rnn(data_dict):
    """
    Apply pre-processing specific to RNN modelling onto the data generated from 'create_data_dict_currency_rnn()'
    """
    return None