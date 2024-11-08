import pandas as pd
import numpy as np
from datetime import timedelta
import eliona.api_client2
from eliona.api_client2.rest import ApiException
from eliona.api_client2.api.data_api import DataApi
import pytz
import os
from sklearn.preprocessing import MinMaxScaler

# Set up configuration for the Eliona API
configuration = eliona.api_client2.Configuration(host=os.getenv("API_ENDPOINT"))
configuration.api_key["ApiKeyAuth"] = os.getenv("API_TOKEN")

# Create an instance of the API client
api_client = eliona.api_client2.ApiClient(configuration)
data_api = DataApi(api_client)


def get_trend_data(asset_id, start_date, end_date):
    asset_id = int(asset_id)
    from_date = start_date.isoformat()
    to_date = end_date.isoformat()
    try:
        print(f"Fetching data for asset {asset_id} from {from_date} to {to_date}")
        result = data_api.get_data_trends(
            from_date=from_date,
            to_date=to_date,
            asset_id=asset_id,
            data_subtype="input",
        )
        print(f"Received {len(result)} data points")
        return result
    except ApiException as e:
        print(f"Exception when calling DataApi->get_data_trends: {e}")
        return None


def fetch_data_in_chunks(asset_id, start_date, end_date):
    all_data = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=15), end_date)
        data_chunk = get_trend_data(asset_id, current_start, current_end)
        if data_chunk:
            all_data.extend(data_chunk)
        current_start = current_end + timedelta(seconds=1)
    return all_data


def convert_to_pandas(data):
    # Dictionary to hold the rows, using the timestamp as the key
    formatted_data = {}

    for entry in data:
        # Extract timestamp and data
        timestamp = entry.timestamp
        data_dict = entry.data

        # If this timestamp already exists, update the existing row
        if timestamp in formatted_data:
            formatted_data[timestamp].update(data_dict)
        else:
            # Create a new row for this timestamp
            formatted_data[timestamp] = data_dict

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(formatted_data, orient="index")

    # Set the index (timestamp) as a proper datetime index
    df.index = pd.to_datetime(df.index, utc=True)

    # Convert the index to the desired timezone (e.g., Europe/Berlin)
    df.index = df.index.tz_convert("Europe/Berlin")

    # **Optional: Sort the DataFrame by index (timestamp)**
    df.sort_index(inplace=True)

    # **Extract the last timestamp**
    last_timestamp = df.index[-1]

    # Reset index to have 'timestamp' as a column
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    return df, last_timestamp


def filter_by_attribute(data, attribute):
    filtered_data = []
    for entry in data:
        if attribute in entry.data:
            filtered_data.append(entry)
    return filtered_data


def fetch_pandas_data(
    asset_id,
    start_date,
    end_date,
    attribute,
):
    # Fetch the data
    data = fetch_data_in_chunks(asset_id, start_date, end_date)
    data = filter_by_attribute(data, attribute)
    df, last_timestamp = convert_to_pandas(data)
    return df, last_timestamp


def prepare_data(data, context_length, forecast_length, taget_attribute):
    """
    Prepares data for training a TensorFlow LSTM model, including scaling.

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame with at least two columns:
        - 'timestamp': The timestamp of each observation.
    - context_length (int): The number of past time steps to use as input.
    - forecast_length (int): The number of time steps ahead to predict.
      The target is the value that is forecast_length steps ahead of the end of the context window.

    Returns:
    - X (np.ndarray): Input features of shape (num_samples, context_length, 1).
    - Y (np.ndarray): Target values of shape (num_samples, 1).
    - scaler (sklearn.preprocessing object): The scaler fitted on the 'brightness' data.
    """

    # Ensure the data is sorted by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Extract the target variable as a numpy array and reshape for scaler
    values = data[taget_attribute].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values).flatten()

    X = []
    Y = []

    # Calculate the number of samples that can be generated
    total_samples = len(scaled_values) - context_length - forecast_length + 1

    # Loop over the dataset to create sequences
    for i in range(total_samples):
        # Extract the input sequence of length 'context_length'
        x = scaled_values[i : i + context_length]
        # Extract the target value that is 'forecast_length' steps ahead
        y_index = i + context_length + forecast_length - 1
        if y_index < len(scaled_values):
            y = scaled_values[y_index]
            X.append(x)
            Y.append(y)
        else:
            break  # Break if the target index is out of bounds

    # Convert lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Reshape X to be [samples, time steps, features] for LSTM input
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # Reshape Y to be [samples, 1]
    Y = Y.reshape((Y.shape[0], 1))

    return X, Y, scaler


def prepare_data_for_forecast(
    data, context_length, forecast_length, scaler, timestep_in_file, target_attribute
):
    """
    Prepares data for forecasting with a TensorFlow LSTM model.

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame with 'timestamp' and target attribute columns.
    - context_length (int): The number of past time steps to use as input.
    - forecast_length (int): The number of time steps ahead to predict.
    - scaler (sklearn.preprocessing object): The fitted scaler for transforming data.
    - timestep_in_file (datetime): The timestamp up to which data has been processed.

    Returns:
    - X (np.ndarray): Input features of shape (num_samples, context_length, 1).
    - Y (np.ndarray): Target values of shape (num_samples, 1).
    - target_timestamps (list): Timestamps corresponding to each target value.
    - X_last (np.ndarray): The last input feature sequence for making the next prediction.
    - new_next_timestamp (datetime): The timestamp for the next prediction.
    """
    # Ensure the data is sorted by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)
    print("data", data.head(20))
    print("timestep_in_file", timestep_in_file)

    # Convert 'timestamp' to datetime if not already
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Identify the last index where timestamp <= timestep_in_file
    past_data = data[data["timestamp"] <= timestep_in_file]
    if not past_data.empty:
        if len(past_data) >= context_length:
            last_past_index = past_data.index[-1]
            # Start from the next index after last_past_index
            new_data = data[data.index > last_past_index]
        else:
            # Not enough past data for context, use all data
            print("Not enough past data for context. Using all available data.")
            new_data = data
    else:
        # No past data, use all data
        print("No past data found. Using all available data.")
        new_data = data

    print("New data to process:", new_data.head(20))

    # Ensure there is data to process
    if new_data.empty:
        print("No new data after the provided timestep.")
        return None, None, None, None, None

    # Extract the target variable as a numpy array and scale
    values = data[target_attribute].values.reshape(-1, 1)
    scaled_values = scaler.transform(values).flatten()

    # Initialize lists
    X = []
    Y = []
    target_timestamps = []

    # Iterate through the new data to create X and Y
    for i in new_data.index:
        current_timestamp = data["timestamp"].iloc[i]
        if current_timestamp <= timestep_in_file:
            continue  # Skip data before or equal to the last processed timestamp

        # Determine the y_index based on forecast_length
        y_index = i + forecast_length - 1
        if y_index >= len(scaled_values):
            continue  # Not enough data for y

        # Ensure there is enough data for context
        if i - context_length < 0:
            print(f"Skipping index {i} due to insufficient context data.")
            continue

        # Extract x and y
        x = scaled_values[i - context_length : i]
        y = scaled_values[y_index]

        # Append to lists
        X.append(x)
        Y.append(y)
        target_timestamps.append(data["timestamp"].iloc[y_index])

    # Convert to numpy arrays
    if not X:
        print("No valid input sequences found.")
        return None, None, None, None, None

    X = np.array(X).reshape((len(X), context_length, 1))
    Y = np.array(Y).reshape((len(Y), 1))
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Sample X:", X[:1])
    print("Sample Y:", Y[:1])

    # Prepare `X_last` for the next prediction
    if len(scaled_values) >= context_length:
        X_last = scaled_values[-context_length:].reshape((1, context_length, 1))
    else:
        X_last = None
        print("Not enough data to prepare X_last.")

    # Calculate `new_next_timestamp`
    if len(data) >= 2:
        timestamp_diff = data["timestamp"].iloc[-1] - data["timestamp"].iloc[-2]
        new_next_timestamp = data["timestamp"].iloc[-1] + (
            timestamp_diff * forecast_length
        )
    else:
        new_next_timestamp = None
        print("Insufficient data for timestamp calculation.")

    return X, Y, target_timestamps, X_last, new_next_timestamp
