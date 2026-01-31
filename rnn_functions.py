import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import RobustScaler

# Dependancy functions:
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

def LSTM_input(df, input_sequence):
    """
    Generate supervised learning sequences from a time-ordered dataset
    for one-step-ahead LSTM forecasting.

    This function converts a time-series DataFrame into sliding input
    sequences (X) and corresponding target values (y), suitable for
    training a many-to-one LSTM model.

    For each sample:
        - X contains `input_sequence` consecutive past observations
        - y contains the immediately following observation

    The function assumes the data is:
        - Ordered in ascending chronological order (oldest → newest)
        - Free of non-numeric columns (e.g. time columns removed)
        - Already scaled or normalised, if required

    Parameters
    ----------
    df : pandas.DataFrame
        Time-series data containing one or more numeric features.
        Shape: (n_samples, n_features).
        The index or original time column is not used by this function.

    input_sequence : int
        Number of past time steps to include in each input sequence
        (i.e. the LSTM lookback window).

    Returns
    -------
    X : numpy.ndarray
        Array of input sequences with shape:
            (n_samples - input_sequence, input_sequence, n_features)

    y : numpy.ndarray
        Array of target values with shape:
            (n_samples - input_sequence, n_features)

    Notes
    -----
    - This function performs one-step-ahead forecasting.
    - Each target value corresponds to the time step immediately
      following its input sequence.
    - The function does not shuffle data and preserves temporal order.
    - The function does not perform any scaling or missing-value handling.

    Example
    -------
    >>> X, y = Sequential_Input_LSTM(df_scaled, input_sequence=28)
    >>> X.shape
    (num_samples, 28, num_features)
    >>> y.shape
    (num_samples, num_features)
    """
    df_np = df.to_numpy()
    X = []
    y = []
    
    for i in range(len(df_np) - input_sequence):
        row = [a for a in df_np[i:i + input_sequence]]
        X.append(row)
        label = df_np[i + input_sequence]
        y.append(label)
        
    return np.array(X), np.array(y)

# To be called in app.py:

def preprocess_all_data_rnn_os(all_data, cc_dict, countries, prefit_scalers: dict = None):
    """
    Create windowed train/test datasets per currency using provided pre-fitted RobustScalers.
    If a scaler for a currency is not provided in `prefit_scalers`, a new RobustScaler will be fit.

    all_data, cc_dict, countries are generated from load_and_process_forex_data() function (called when app.py is first run)

    Returns: training_dict, testing_dict
    """
    data_dict = create_data_dict_currency_rnn(all_data, countries, cc_dict)

    training_dict = {}
    testing_dict = {}

    for country in countries:
        currency = cc_dict[country]

        df = data_dict[currency].copy()
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        test_df = df.tail(60).reset_index(drop=True)
        train_df = df.iloc[:-60].reset_index(drop=True)

        # drop Date column before scaling if present
        if "Date" in train_df.columns:
            X_train_raw = train_df.drop(columns=["Date"]).copy()
        else:
            X_train_raw = train_df.copy()
        if "Date" in test_df.columns:
            X_test_raw = test_df.drop(columns=["Date"]).copy()
        else:
            X_test_raw = test_df.copy()

        # use provided pre-fitted scaler if available, otherwise fit a new one
        if prefit_scalers is not None and currency in prefit_scalers and prefit_scalers[currency] is not None:
            scaler = prefit_scalers[currency]
            X_train_scaled = scaler.transform(X_train_raw.values)
            X_test_scaled = scaler.transform(X_test_raw.values)
        else:
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw.values)
            X_test_scaled = scaler.transform(X_test_raw.values)

        train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns, index=X_train_raw.index)
        test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_raw.columns, index=X_test_raw.index)

        train_windowed_X, train_windowed_y = LSTM_input(train_scaled_df, 10)
        test_windowed_X, test_windowed_y = LSTM_input(test_scaled_df, 10)

        training_dict[currency] = {"X": train_windowed_X, "y": train_windowed_y}
        testing_dict[currency] = {"X": test_windowed_X, "y": test_windowed_y}

    return training_dict, testing_dict

def load_saved_rnn_scalers(saved_files_dict):
    """
    Load saved RNN scalers from pickle files.
    
    Parameters:
    -----------
    saved_files_dict : dict
        Dictionary mapping currency names to their saved file paths
    
    Returns:
    --------
    dict: Dictionary containing loaded scalers with currency names as keys
    """
    
    loaded_rnn_scalers = {}
    
    for currency_name, filepath in saved_files_dict.items():
        try:
            with open(filepath, 'rb') as f:
                scaler = pickle.load(f)
            loaded_rnn_scalers[currency_name] = scaler
            print(f"Successfully loaded {currency_name} scaler from {filepath}")
        except FileNotFoundError:
            print(f"Error: Could not find file {filepath} for {currency_name}")
        except Exception as e:
            print(f"Error loading {currency_name} scaler: {str(e)}")
    
    return loaded_rnn_scalers

def load_saved_rnn_models(saved_files_dict):
    """
    Load saved RNN models from pickle files.
    
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

def incremental_forecast_to_df(
    model,
    train_X,
    train_y,
    lookback=10,
    horizon=60,
    scaler=None,
    start_step=1,
    column_name="forecast",
):
    """
    Incremental (recursive) forecasting for a univariate (or multivariate) windowed TF RNN model (single currency)

    Requires:
      - train_X: windowed 3D array (n_windows, lookback, n_features). The last window is used as history.
      - train_y: corresponding targets for train_X (used for basic validation).
    Outputs:
      - pd.DataFrame indexed by integer steps [start_step .. start_step+horizon-1] with column `column_name`.

    Example Usage:
    
    # Forecast 60 days for Australian Dollar
    aus_currency = cc_dict['AUSTRALIA']
    train_X_a = training_dict[aus_currency]['X']
    train_y_a = training_dict[aus_currency]['y']
    aus_scaler = scaler_dict[aus_currency]

    aus_forecast_df = incremental_forecast_to_df(
        model=australian_model,
        train_X=train_X_a,
        train_y=train_y_a,
        lookback=lookback_length,
        horizon=60,
        scaler=aus_scaler,
        column_name='AUSTRALIAN_DOLLAR_US$'
    )
    """
    # --- prepare window from train_X ---
    if train_X is None:
        raise ValueError("`train_X` (windowed) must be provided.")
    hx = np.asarray(train_X, dtype=np.float32)
    if hx.ndim != 3:
        raise ValueError("`train_X` must be windowed with shape (n_windows, lookback, n_features).")
    # validate train_y shape matches train_X windows
    hy = np.asarray(train_y, dtype=np.float32)
    if hy.ndim == 1:
        hy = hy.reshape(-1, 1)
    if hy.shape[0] != hx.shape[0]:
        raise ValueError("`train_y` must have same number of rows as `train_X` windows.")
    window = hx[-1:].astype(np.float32)  # shape (1, lookback, n_features)

    # --- validations ---
    if horizon < 1:
        raise ValueError("`horizon` must be >= 1.")
    if lookback < 1:
        raise ValueError("`lookback` must be >= 1.")
    if window.shape[1] != lookback:
        raise ValueError(f"Window has lookback={window.shape[1]} but function expected lookback={lookback}.")

    n_features = window.shape[2]
    preds_scaled = np.zeros((horizon, n_features), dtype=np.float32) if n_features > 1 else np.zeros((horizon, 1), dtype=np.float32)

    for t in range(horizon):
        yhat = model.predict(window, verbose=0)
        yhat = np.asarray(yhat).reshape(yhat.shape[0], -1)  # (1, k) where k may be 1
        # if model predicts full feature vector, take that; else assume scalar and expand
        if yhat.shape[1] == 1 and n_features != 1:
            row = np.zeros((n_features,), dtype=np.float32)
            row[0] = np.float32(yhat[0, 0])
        else:
            row = yhat[0].astype(np.float32)
        preds_scaled[t, :] = row

        # update window: drop oldest timestep, append prediction as new last timestep
        new_step = row.reshape(1, 1, -1)  # (1,1,n_features)
        window = np.concatenate([window[:, 1:, :], new_step], axis=1)

    # --- inverse transform if scaler provided ---
    if scaler is not None:
        preds = scaler.inverse_transform(preds_scaled.reshape(-1, preds_scaled.shape[1]))
    else:
        preds = preds_scaled

    # --- build DataFrame with integer step index ---
    steps = np.arange(start_step, start_step + horizon, dtype=int)
    if preds.shape[1] == 1:
        out = pd.DataFrame({column_name: preds.reshape(-1)}, index=steps)
    else:
        cols = [f"{column_name}_{i}" for i in range(preds.shape[1])]
        out = pd.DataFrame(preds, index=steps, columns=cols)
    out.index.name = "step"
    return out