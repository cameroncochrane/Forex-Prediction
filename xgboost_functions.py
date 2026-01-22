import pandas as pd
import numpy as np

def extract_currency_df_xgboost(data, currency):
    """
    Extract a specific country's data from the main (unprocessed) dataframe.
    
    Parameters:
    data (DataFrame): The main dataframe containing all countries' data
    country_name (str): The name of the country to extract data for
    
    Returns:
    DataFrame: A dataframe containing only the specified country's data with date column
    """
    country_data = data[['Date',currency]].copy()
    
    # Ensure date column is included and properly formatted
    
    return country_data

def create_data_dict_currency_xgboost(data,countries,currency_dict):
    """
    Uses extract_currency_df_xgboost() to create individual dataframes for each currency and places
    them in a dictionary of dataframes, ready for pre-processing for XGBoost modelling.
    """


    data_dict = {}
    for country in countries:
        data_dict[currency_dict[country]] = extract_currency_df_xgboost(data,currency_dict[country])
    
    return data_dict

# Processing function:
def prepare_single_time_series_indexed_xgboost(
    df: pd.DataFrame,
    currency: str,
    horizon: int = 60, #Test set size (in rows)
    lags: tuple = (1, 2, 3, 5, 7, 14, 21, 28, 35, 42, 49, 56),
    rolling_windows: tuple = (7, 14, 28, 56),
) -> dict:
    """
    Prepare a univariate daily time series for XGBoost training using an index-based approach on a single currency.
    Non-random splitting used

    df = [Date/Index, Price/Currency]
    currency = 'EURO/US$','UNITED KINGDOM POUND/US$' etc
            

    Assumptions:
      - Each row is separated by exactly one day
      - Data is already sorted chronologically
      - Model predicts next-day price (t+1)
      - 60-day forecast achieved via recursive prediction

    Train-test split:
      - Last `horizon` rows used as test set

    Returns:
      dict containing:
        X_train, y_train, X_test, y_test, feature_cols, df_features
    """

    if currency not in df.columns:
        raise ValueError(f"Column '{currency}' not found in dataframe.")

    data = df[[currency]].copy().reset_index(drop=True)

    # Create simple integer time index
    data["t"] = np.arange(len(data))

    # Target: next-day price
    data["y"] = data[currency].shift(-1)

    # Lag features
    for lag in lags:
        data[f"lag_{lag}"] = data[currency].shift(lag)

    # Rolling statistics (past-only)
    shifted = data[currency].shift(1)
    for w in rolling_windows:
        data[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        data[f"roll_std_{w}"]  = shifted.rolling(w).std()
        data[f"roll_min_{w}"]  = shifted.rolling(w).min()
        data[f"roll_max_{w}"]  = shifted.rolling(w).max()

    # Momentum features
    data["diff_1"] = data[currency].diff(1)
    data["pct_change_1"] = data[currency].pct_change(1)

    # Feature columns (exclude raw price + target)
    exclude = {currency, "y"}
    feature_cols = [c for c in data.columns if c not in exclude]

    # Drop rows with NaNs caused by shifting/rolling
    data_model = data.dropna().copy()

    if len(data_model) <= horizon:
        raise ValueError("Not enough data after feature creation for the chosen horizon.")

    # Time-based split (no shuffling)
    train_df = data_model.iloc[:-horizon]
    test_df = data_model.iloc[-horizon:]

    X_train = train_df[feature_cols]
    y_train = train_df["y"]
    X_test = test_df[feature_cols]
    y_test = test_df["y"]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "df_features": data_model
    }

def process_all_xgboost(raw_gradient_data):
    """
    Apply the prepare_single_time_series_indexed_xgboost() function to a dictionary of datframes containing all 
    currencies (created from create_data_dict_currency_xgboost() function, with structure {currency:df}
    """
    processed_xgb_data = {}

    for currency, df in raw_gradient_data.items():
        # Get the currency column name (second column after Date)
        price_col = df.columns[1]
        
        # Apply the XGBoost preparation function (for a single currency)
        processed_xgb_data[currency] = prepare_single_time_series_indexed_xgboost(
            df=df,
            currency=price_col,
            horizon=60,
            lags=(1, 2, 3, 5, 7, 14, 21, 28, 35, 42, 49, 56),
            rolling_windows=(7, 14, 28, 56)
        )
    
    return processed_xgb_data