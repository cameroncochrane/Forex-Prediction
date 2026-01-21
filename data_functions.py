# Functions to import and split the Forex data:

from pathlib import Path
import pandas as pd

import statsmodels
from statsmodels.tsa.arima.model import ARIMA

from arima_functions import *


# Functions:

# This function also generates country_currency and country names.
def load_and_process_forex_data():
    """
    Load and process forex exchange rate data from the 'Foreign_Exchange_Rates.xlsx' Excel file.
    
    Returns:
    tuple: (processed_dataframe, country_currency_dict, country_names)
    """
     # Import the data
    cwd_path = Path.cwd()
    data_dir = Path('data')
    d_file_path = cwd_path / data_dir / 'Foreign_Exchange_Rates.xlsx'
    raw_data = pd.read_excel(d_file_path)

    # Drop the 'Unnamed: 0' column
    data = raw_data[['Time Serie', 'EURO AREA - EURO/US$', 'UNITED KINGDOM - UNITED KINGDOM POUND/US$', 'JAPAN - YEN/US$', 'CHINA - YUAN/US$', 'AUSTRALIA - AUSTRALIAN DOLLAR/US$']]
    
    # Drop rows containing 'ND' values
    data = data[~data.isin(['ND']).any(axis=1)]
    
    # Extract country names from column headers
    country_names = []
    for col in data.columns[1:]:  # Skip 'Time Serie' column
        if '/' in col:
            # Extract everything before the ' - ' or before '/US$'
            if ' - ' in col:
                country = col.split(' - ')[0]
            else:
                # For cases where there's no ' - ', extract before '/US$'
                country = col.split('/')[0]
            country_names.append(country)
    
    # Clean column names by removing text before '-'
    new_columns = []
    for col in data.columns:
        if ' - ' in col:
            # Split by ' - ' and take the part after the dash
            new_name = col.split(' - ', 1)[1]
            new_columns.append(new_name)
        else:
            # Keep the original name if no dash is found (like 'Time Serie')
            new_columns.append(col)
    
    # Apply the new column names
    data.columns = new_columns
    
    # Convert 'Time Serie' column to datetime with d/m/y format
    data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d/%m/%Y')
    
    # Rename 'Time Serie' column to 'Date'
    data = data.rename(columns={'Time Serie': 'Date'})
    
    # Convert all columns except 'Date' to float
    for col in data.columns[1:]:  # Skip the first column 'Date'
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Create a dictionary mapping country names to currency names
    country_currency_dict = {}
    currency_names = data.columns
    for i, country in enumerate(country_names):
        # Get the corresponding currency name (skip 'Date' column)
        currency = currency_names[i + 1]  # +1 to skip 'Date' column
        country_currency_dict[country] = currency
    
    return data, country_currency_dict, country_names

def train_test_split_timeseries_arima(series, test_size=60):
    """
    Split a time series (pandas Series) into train and test sets.
    
    Parameters:
    -----------
    series : pd.Series
        The time series data (already ordered)
    test_size : int, default=60
        Number of observations to use for test set (from the end)
    
    Returns:
    --------
    tuple: (train_series, test_series)
        Training and testing series
    """
    # Calculate split index
    split_index = len(series) - test_size
    
    # Split the data
    train_series = series.iloc[:split_index].copy()
    test_series = series.iloc[split_index:].copy()
    
    return train_series, test_series

def train_test_split_dataframe_arima(df, test_size=60):
    """
    Split a dataframe with time series data into train and test sets, with test size defining how many rows from the end the test set size.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing time series data
    test_size : int, default=60
        Number of observations to use for test set (from the end)

    Returns:
    --------
    tuple: (train_df, test_df)
        Training and testing dataframes
    """
    # Calculate split index
    split_index = len(df) - test_size
    
    # Split the data
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    return train_df, test_df
