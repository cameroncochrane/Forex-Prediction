import pandas as pd
import numpy as np
import pickle

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

# Recursive forecasting functions:

def make_recursive_forecast_xgboost(
    xgb_model,
    df_features: pd.DataFrame,
    currency_col: str,
    feature_cols: list[str],
    forecast_length: int,
    lags: tuple = (1, 2, 3, 5, 7, 14, 21, 28, 35, 42, 49, 56),
    rolling_windows: tuple = (7, 14, 28, 56),
) -> pd.DataFrame:
    """
    True recursive forecaster matching prepare_single_time_series_indexed_xgboost() exactly.

    Inputs
    ------
    df_features:
        Should be the 'df_features' returned by prepare_single_time_series_indexed_xgboost().
        Must contain at least columns: [currency_col, 't', ...features...]
    currency_col:
        The raw price column name used during training (e.g., 'EURO/US$').
    feature_cols:
        The feature column list returned by prepare_single_time_series_indexed_xgboost() (includes 't').
    forecast_length:
        Number of steps to forecast recursively.

    Returns
    -------
    DataFrame indexed by step (1..forecast_length) with forecast_price and (optional) t.

    Example
    --------
    bundle = processed_xgb_data['EURO/US$']  #Generated via: processed_xgb_data = process_all_xgboost(raw_gradient_data)

    forecast_df = make_recursive_forecast_xgboost(
        xgb_model=euro_model,
        df_features=bundle["df_features"],
        currency_col=bundle["df_features"].columns[0],  # or the known currency column name
        feature_cols=bundle["feature_cols"],
        forecast_length=60
    )

    """

    if forecast_length <= 0:
        raise ValueError("forecast_length must be > 0.")
    if currency_col not in df_features.columns:
        raise ValueError(f"currency_col '{currency_col}' not found in df_features.")
    if any(c not in df_features.columns for c in feature_cols):
        missing = [c for c in feature_cols if c not in df_features.columns]
        raise ValueError(f"feature_cols contain columns not in df_features: {missing}")

    # Work from the end of the feature-engineered dataset
    data = df_features[[currency_col]].copy().reset_index(drop=True)
    data.index.name = "row"

    # We maintain an explicit integer time index 't' just like in your training function
    # Start t from the last known t in df_features (or from len(data)-1 if not present)
    if "t" in df_features.columns:
        last_t = int(df_features["t"].iloc[-1])
    else:
        last_t = len(data) - 1

    def _build_full_feature_frame(price_df: pd.DataFrame, start_t: int) -> pd.DataFrame:
        """Recreate features exactly as in prepare_single_time_series_indexed_xgboost()."""
        df = price_df.copy()
        df["t"] = np.arange(start_t - len(df) + 1, start_t + 1)  # keep 't' consistent/increasing

        # Target (not used for prediction, but keep for parity)
        df["y"] = df[currency_col].shift(-1)

        # Lag features
        for lag in lags:
            df[f"lag_{lag}"] = df[currency_col].shift(lag)

        # Rolling stats (past-only)
        shifted = df[currency_col].shift(1)
        for w in rolling_windows:
            df[f"roll_mean_{w}"] = shifted.rolling(w).mean()
            df[f"roll_std_{w}"]  = shifted.rolling(w).std()
            df[f"roll_min_{w}"]  = shifted.rolling(w).min()
            df[f"roll_max_{w}"]  = shifted.rolling(w).max()

        # Momentum
        df["diff_1"] = df[currency_col].diff(1)
        df["pct_change_1"] = df[currency_col].pct_change(1)

        return df

    forecasts = []
    current_last_t = last_t

    for step in range(1, forecast_length + 1):
        # Build features on the evolving price history
        feat_df = _build_full_feature_frame(data, start_t=current_last_t)

        # Find latest row with complete features (no NaNs in feature_cols)
        valid = feat_df.dropna(subset=feature_cols)
        if valid.empty:
            raise RuntimeError(
                "No valid feature row available (all NaN). "
                "Usually means not enough history for lags/rolls."
            )

        latest = valid.iloc[-1:]
        X_latest = latest[feature_cols]

        # Predict next-day PRICE
        y_hat = float(xgb_model.predict(X_latest)[0])

        # Append predicted price as the next row
        data = pd.concat(
            [data, pd.DataFrame({currency_col: [y_hat]})],
            ignore_index=True
        )

        current_last_t += 1  # advance time index

        forecasts.append({"step": step, "t": current_last_t, "forecast_price": y_hat})

    ## NEEDS MODIFYING SO THE RETURNED DATAFRAME IS 2 COLUMNS: 'forecast_period' and 'currency_predicted_price', where forecast period is from 1 to forecast_length ##

    return pd.DataFrame(forecasts).set_index("step")

def load_saved_gb_models(saved_files_dict):
    """
    Load saved XGBoost models from pickle files.
    
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