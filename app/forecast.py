import time
import pytz
import os
from datetime import datetime, timedelta
from fetch_and_format_data import fetch_and_format_data
from write_into_eliona import write_into_eliona
import argparse
import pandas as pd
import tensorflow as tf
import numpy as np


@tf.function
def train_step(model, x, y):
    return model.train_on_batch(x, y)


def save_latest_timestamp(timestamp, file_path):
    with open(file_path, "w") as file:
        file.write(timestamp.isoformat())


def load_latest_timestamp(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            timestamp_str = file.read()
            return datetime.fromisoformat(timestamp_str)
    return None


def main():
    parser = argparse.ArgumentParser(description="LSTM Training and Prediction Script")
    parser.add_argument("--asset_id", type=int, default=14745, help="Asset ID")
    parser.add_argument(
        "--forecast_length", type=int, default=1, help="Forecast length"
    )
    parser.add_argument(
        "--target_column", type=str, default="supply_temperature", help="Target column"
    )
    parser.add_argument(
        "--feature_columns",
        type=str,
        nargs="+",
        default=["supply_temperature"],
        help="Feature columns",
    )

    args = parser.parse_args()

    asset_id = args.asset_id
    tz = pytz.timezone("Europe/Berlin")

    forecast_length = args.forecast_length
    target_column = args.target_column
    feature_columns = args.feature_columns
    model_filename = f"LSTM_model_{asset_id}_{forecast_length}.keras"
    timestamp_file = f"LSTM_model_{asset_id}_{forecast_length}_latest_timestamp.txt"
    context_length_file_name = (
        f"LSTM_model_{asset_id}_{forecast_length}_context_length.txt"
    )
    batch_size = 1  # Setting batch size to 1 for stateful LSTM

    if os.path.exists(model_filename):
        print(f"Loading existing model from {model_filename}")
        model = tf.keras.models.load_model(model_filename)
        context_length = int(open(context_length_file_name).read()) * 3
        new_end_date = datetime.now(tz)
        new_start_date = new_end_date - timedelta(days=3)
        new_X, new_y, new_target_timestamps, new_X_last, new_next_timestamp = (
            fetch_and_format_data(
                asset_id,
                new_start_date,
                new_end_date,
                context_length,
                forecast_length,
                target_column,
                feature_columns,
            )
        )
        timestep_in_file = load_latest_timestamp(timestamp_file)
        # Check if there is new data by comparing timestamps
        print("Timestep in file:", timestep_in_file)
        print("New data timestamps:", new_target_timestamps[-1])
        if new_target_timestamps[-1] > timestep_in_file:
            print("New data found, updating state...")

            # Filter new data that is not in the existing data
            last_known_timestamp = timestep_in_file
            new_data_mask = pd.Series(new_target_timestamps) > last_known_timestamp
            new_X = new_X[new_data_mask]
            new_y = new_y[new_data_mask]
            new_target_timestamps = pd.Series(new_target_timestamps)[new_data_mask]

            # Use the new data to update the state
            print("new_y", new_y)

            def train_step(model, x, y):
                return model.train_on_batch(x, y)

            for i in range(len(new_X)):
                x = np.expand_dims(new_X[i], axis=0)
                y = np.expand_dims(new_y[i], axis=0)
                train_step(model, x, y)

            # Make a prediction with the updated state
            print("new_X_last", new_X_last)
            next_prediction = model.predict(new_X_last, batch_size=batch_size)
            print("Next prediction:", next_prediction[0][0])
            print("Predicted next timestamp:", new_next_timestamp)

            # Write prediction into Eliona
            write_into_eliona(
                asset_id,
                new_next_timestamp,
                next_prediction[0][0],
                target_column,
                forecast_length,
            )

            # Update target timestamps with new data
            target_timestamps = pd.Series(new_target_timestamps).tolist()

            # Save the latest timestamp
            save_latest_timestamp(target_timestamps[-1], timestamp_file)
            model.save(model_filename)


if __name__ == "__main__":
    while True:
        main()
        time.sleep(60)
