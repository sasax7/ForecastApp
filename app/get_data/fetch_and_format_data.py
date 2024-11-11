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

    # Reset index to have 'timestamp' as a column
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    return df


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
    df = convert_to_pandas(data)
    return df


def prepare_data(data, context_length, forecast_length, target_attribute):
    """
    Prepares data for training a TensorFlow LSTM model, including scaling.
    """
    # Ensure the data is sorted by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Extract the target variable as a numpy array and reshape for scaler
    values = data[target_attribute].values.reshape(-1, 1)

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

    # Extract the last Y's timestamp
    last_timestamp = data["timestamp"].iloc[len(data) - forecast_length]
    last_timestamp = pd.to_datetime(last_timestamp)

    return X, Y, scaler, last_timestamp


def prepare_data_for_forecast(
    data, context_length, forecast_length, scaler, last_timestamp, target_attribute
):
    # Ensure the data is sorted by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)
    print("Data Head:\n", data.head(20))
    print("Last Timestamp from training data:", last_timestamp)

    # Convert 'timestamp' to datetime if not already
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Use the loaded scaler to transform values
    values = data[target_attribute].values.reshape(-1, 1)
    scaled_values = scaler.transform(values).flatten()

    # Find the index corresponding to last_timestamp
    try:
        last_index = data[data["timestamp"] == last_timestamp].index[0]
        last_index = last_index + 1  # Move to the next index
    except IndexError:
        # If exact match not found, find the closest timestamp after last_timestamp
        indices = data[data["timestamp"] > last_timestamp].index
        print("No exact match found. Closest timestamps found at indices:", indices)
        if len(indices) == 0:
            print("No new data available after the last timestamp.")
            return None, None, None, None
        last_index = indices[0]

    # Start index to include context_length steps before last_index
    start_index = last_index - context_length + 1
    if start_index < 0:
        start_index = 0

    # Prepare sequences
    X_new = []
    target_timestamps = []

    for i in range(start_index, len(scaled_values) - context_length + 1):
        x = scaled_values[i : i + context_length]
        X_new.append(x)
        timestamp = data["timestamp"].iloc[i + context_length - 1]
        target_timestamps.append(timestamp)

    if not X_new:
        print("No valid input sequences found after filtering.")
        return None, None, None, None

    X_new = np.array(X_new).reshape((len(X_new), context_length, 1))

    # Separate X_new into X_update and X_last
    if len(X_new) > 1:
        X_update = X_new[:-1]
        X_last = X_new[-1].reshape((1, context_length, 1))
        last_y_timestamp_new = target_timestamps[-2]
    else:
        X_update = np.empty((0, context_length, 1))
        X_last = X_new[-1].reshape((1, context_length, 1))
        last_y_timestamp_new = target_timestamps[-1]

    # Calculate `new_next_timestamp`
    if len(data) >= 2:
        timestamp_diff = data["timestamp"].iloc[-1] - data["timestamp"].iloc[-2]
        new_next_timestamp = (
            data["timestamp"].iloc[-1] + timestamp_diff * forecast_length
        )
    else:
        new_next_timestamp = None
        print("Insufficient data for timestamp calculation.")

    return X_update, X_last, new_next_timestamp, last_y_timestamp_new
