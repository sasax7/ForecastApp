import time
import pytz
import os
from datetime import datetime
from fetch_and_format_data import fetch_and_format_data, fetch_pandas_data
from train import compile_and_train_model, create_lstm_tuned_model
from add_forecast_attributes import add_forecast_attributes
import pandas as pd
from tensorflow.keras.models import load_model


def save_latest_timestamp(timestamp, file_path):
    with open(file_path, "w") as file:
        file.write(timestamp.isoformat())


def load_latest_timestamp(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            timestamp_str = file.read()
            return datetime.fromisoformat(timestamp_str)
    return None


def save_datalength(datasize, file_path):
    with open(file_path, "w") as file:
        file.write(str(datasize))


def load_datalength(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            datasize_str = file.read()
            try:
                return int(datasize_str)
            except ValueError:
                print("Error: Could not convert datasize to integer")
                return None
    return None


def train_and_retrain(
    asset_id,
    start_date_str,
    forecast_length,
    target_column,
    feature_columns,
    sleep_time,
):
    tz = pytz.timezone("Europe/Berlin")
    start_date = tz.localize(datetime.strptime(start_date_str, "%Y-%m-%d"))
    forecast_name_suffix = f"_forecast_{forecast_length}"
    model_filename = f"LSTM_model_{asset_id}_{forecast_length}.keras"
    timestamp_file = f"LSTM_model_{asset_id}_{forecast_length}_latest_timestamp.txt"
    datasize_file = f"LSTM_model_{asset_id}_{forecast_length}_datasize.txt"
    tuning_project_name = f"LSTM_model_{asset_id}_{forecast_length}__tuning"
    context_length_file_name = (
        f"LSTM_model_{asset_id}_{forecast_length}_context_length.txt"
    )
    batch_size = 1  # Setting batch size to 1 for stateful LSTM

    # Continuous loop for periodic retraining
    while True:
        end_date = tz.localize(datetime.now())

        if os.path.exists(model_filename):
            print("Model exists")
            context_length = load_datalength(context_length_file_name) * 3
            X, y, target_timestamps, X_last, next_timestamp = fetch_and_format_data(
                asset_id,
                start_date,
                end_date,
                context_length,
                forecast_length,
                target_column,
                feature_columns,
            )

            target_timestamps = pd.Series(target_timestamps).tolist()
            data_length = load_datalength(datasize_file)

            print("Data length", data_length * 1.15)
            print("Length X", len(X))
            if len(X) > data_length * 1.15:
                model = load_model(model_filename)
                model = compile_and_train_model(
                    model,
                    X,
                    y,
                    batch_size,
                    epochs=100000,
                    validation_split=0.2,
                    patience=20,
                )
                model.save(model_filename)
                save_latest_timestamp(target_timestamps[-1], timestamp_file)
                save_datalength(len(X), datasize_file)
                print(f"Model saved as {model_filename}")
        else:
            print("Model does not exist")

            df = fetch_pandas_data(asset_id, start_date, end_date)
            print(df.head())
            model, X, last_timestamp, context_length = create_lstm_tuned_model(
                df,
                batch_size,
                forecast_length,
                target_column,
                feature_columns,
                project_name=tuning_project_name,
                max_trials=100,
                patience=20,
                model_save_path=model_filename,
                context_length_file_name=context_length_file_name,
                timestamp_file=timestamp_file,
            )
            model.save(model_filename)
            print("latest timestamp", last_timestamp)
            save_latest_timestamp(last_timestamp, timestamp_file)
            print("Length X", len(X))
            save_datalength(len(X), datasize_file)
            print("context_length", context_length)
            save_datalength(context_length, context_length_file_name)

            print("train and compile model on all data")
            X, y, target_timestamps, X_last, next_timestamp = fetch_and_format_data(
                asset_id,
                start_date,
                end_date,
                context_length,
                forecast_length,
                target_column,
                feature_columns,
            )

            model = compile_and_train_model(
                model,
                X,
                y,
                batch_size,
                epochs=100000,
                validation_split=0.2,
                patience=20,
            )
            model.save(model_filename)
            print("latest timestamp", last_timestamp)
            save_latest_timestamp(last_timestamp, timestamp_file)
            print("Length X", len(X))
            save_datalength(len(X), datasize_file)
            print("context_length", context_length)
            save_datalength(context_length, context_length_file_name)

        # Wait for the specified sleep time before running again
        print(f"Sleeping for {sleep_time} seconds before next retraining cycle...")
        time.sleep(sleep_time)
