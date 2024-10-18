import pandas as pd
import numpy as np
from datetime import timedelta
import eliona.api_client2
from eliona.api_client2.rest import ApiException
from eliona.api_client2.api.data_api import DataApi
import pytz
import os

# Set up configuration for the Eliona API
configuration = eliona.api_client2.Configuration(host=os.getenv("API_ENDPOINT"))
configuration.api_key["ApiKeyAuth"] = os.getenv("API_TOKEN")

# Create an instance of the API client
api_client = eliona.api_client2.ApiClient(configuration)
data_api = DataApi(api_client)


def get_trend_data(asset_id, start_date, end_date):
    asset_id = int(asset_id)
    from_date = start_date.astimezone(pytz.utc).isoformat()
    to_date = end_date.astimezone(pytz.utc).isoformat()
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
    df.index = pd.to_datetime(df.index)

    # Optional: convert the index to a specific timezone (e.g., Europe/Berlin)
    df.index = df.index.tz_convert("Europe/Berlin")

    # Reset index to have 'timestamp' as a column
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    return df


def add_time_features(df):
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week
    return df


def format_lstm_data_with_time_features(
    df, context_length, forecast_length, target_column, feature_columns=None
):
    # Ensure the data is sorted by timestamp, timezone-aware, and drop duplicate timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )

    df.sort_values(by="timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", inplace=True)

    # If feature_columns is None, use all columns except timestamp and target_column
    if feature_columns is None:
        feature_columns = df.drop(columns=["timestamp", target_column]).columns.tolist()

    # Extract the values and timestamps
    feature_values = df[feature_columns].values
    target_values = df[target_column].values
    timestamps = df["timestamp"]

    X, y, target_timestamps = [], [], []

    for i in range(len(df) - context_length - forecast_length + 1):
        X.append(feature_values[i : i + context_length])
        y.append(target_values[i + context_length + forecast_length - 1])
        target_timestamps.append(
            timestamps.iloc[i + context_length + forecast_length - 1]
        )

    # Capture the last sequence for future prediction
    X_last = feature_values[-context_length:]

    # Calculate the time differences for the last 100 time steps
    time_diffs = (timestamps.diff().dropna()[-100:]).value_counts().idxmax()

    # Predict the next timestamp
    next_timestamp = timestamps.iloc[-1] + forecast_length * time_diffs

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        target_timestamps,
        np.expand_dims(X_last, axis=0).astype(np.float32),
        next_timestamp,
    )


def fetch_and_format_data(
    asset_id,
    start_date,
    end_date,
    context_length,
    forecast_length,
    target_column,
    feature_columns=None,
):
    # Fetch the data
    data = fetch_data_in_chunks(asset_id, start_date, end_date)

    df = convert_to_pandas(data)

    # Add time features
    df = add_time_features(df)

    # Format the data for LSTM input
    X, y, target_timestamps, X_last, next_timestamp = (
        format_lstm_data_with_time_features(
            df, context_length, forecast_length, target_column, feature_columns
        )
    )

    return X, y, target_timestamps, X_last, next_timestamp


def fetch_pandas_data(
    asset_id,
    start_date,
    end_date,
):
    # Fetch the data
    data = fetch_data_in_chunks(asset_id, start_date, end_date)
    print(data[:10])
    df = convert_to_pandas(data)
    return df
