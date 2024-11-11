from datetime import datetime, timedelta
import pytz
import os
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from app.get_data.api_calls import (
    load_latest_timestamp,
    save_latest_timestamp,
    load_contextlength,
    load_scaler,
    loadState,
    saveState,
)
from app.get_data.fetch_and_format_data import (
    fetch_pandas_data,
    prepare_data_for_forecast,
)
from app.data_to_eliona.write_into_eliona import write_into_eliona


@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
loss_fn = tf.keras.losses.MeanSquaredError()


def forecast(
    asset_details,
    asset_id,
    sleep_time=3600,
):
    # Initialize database connection and ORM models here
    db_url = os.getenv("CONNECTION_STRING")
    db_url_sql = db_url.replace("postgres", "postgresql")
    DATABASE_URL = db_url_sql
    engine = create_engine(DATABASE_URL)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )
    id = asset_details["id"]
    forecast_length = asset_details["forecast_length"]
    target_column = asset_details["target_attribute"]
    tz = pytz.timezone("Europe/Berlin")
    model_filename = f"LSTM_model_{asset_id}_{target_column}_{forecast_length}.keras"
    batch_size = 1  # Setting batch size to 1 for stateful LSTM
    timestamp_diff_buffer = timedelta(days=1.5)

    while True:
        if os.path.exists(model_filename):
            print(f"Loading existing model from {model_filename}")
            model = tf.keras.models.load_model(model_filename)
            loadState(SessionLocal, Asset, model, asset_details)
            # Load the scaler
            scaler = load_scaler(SessionLocal, Asset, asset_details)

            timestep_in_file = load_latest_timestamp(SessionLocal, Asset, asset_details)
            timestep_in_file = datetime.fromisoformat(timestep_in_file)
            context_length = load_contextlength(SessionLocal, Asset, asset_details)

            new_end_date = datetime.now(tz)
            print("timestep_in_file", timestep_in_file)
            new_start_date = (timestep_in_file - timestamp_diff_buffer * 2).astimezone(
                tz
            )

            print("new_start_date", new_start_date)
            print("timestamp_diff_buffer", timestamp_diff_buffer)

            df = fetch_pandas_data(
                asset_id, new_start_date, new_end_date, target_column
            )

            if df.empty:
                print("No data fetched, skipping iteration.")
                time.sleep(sleep_time)
                continue

            X_update, X_last, new_next_timestamp, last_y_timestamp = (
                prepare_data_for_forecast(
                    df,
                    context_length,
                    forecast_length,
                    scaler,
                    timestep_in_file,
                    asset_details["target_attribute"],
                )
            )
            print("Prepared data")

            if X_update is None and X_last is None:
                print("No new X sequences to process. Sleeping...")
                time.sleep(sleep_time)
                continue

            if len(X_update) > 0:
                print(f"Updating model's state with {len(X_update)} new X sequences.")
                print("First sequences from forecasting data:", X_update[:3])
                for i in range(len(X_update)):
                    x = X_update[i].reshape(
                        (1, context_length, 1)
                    )  # Shape: (1, context_length, 1)
                    _ = model.predict(
                        x, batch_size=batch_size
                    )  # Perform prediction to update state
                print("Model's state updated with new X sequences.")

            # Forecast the next y using X_last
            if X_last is not None:
                print("last x sequence:", X_last)
                print("Forecasting the next value using the latest X sequence.")
                next_prediction_scaled = model.predict(X_last, batch_size=batch_size)
                next_prediction = scaler.inverse_transform(next_prediction_scaled)
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

                # Update the latest timestamp to last_y_timestamp
                save_latest_timestamp(
                    SessionLocal, Asset, last_y_timestamp, tz, asset_details
                )

                print("Latest timestamp updated.")
            else:
                print("X_last is None. Skipping forecasting.")

            # Save the updated model after processing
            model.save(model_filename)
            saveState(SessionLocal, Asset, model, asset_details)
            print(f"Model saved to {model_filename}.")
        else:
            print(f"Model {model_filename} does not exist. Skipping iteration.")

        print(f"Sleeping for {sleep_time} seconds before the next forecast...")
        time.sleep(sleep_time)
