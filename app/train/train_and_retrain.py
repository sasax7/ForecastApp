import time
import pytz
import os
from datetime import datetime
from app.get_data.fetch_and_format_data import fetch_pandas_data
from app.train.train import train_lstm_model
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
)

from app.get_data.api_calls import (
    save_datalength,
    save_latest_timestamp,
    load_contextlength,
    load_datalength,
    save_scaler,
)


def train_and_retrain(
    asset_details,
    asset_id,
    sleep_time,
):
    id = asset_details["id"]
    forecast_length = asset_details["forecast_length"]
    target_column = asset_details["target_attribute"]
    tz = pytz.timezone("Europe/Berlin")
    start_date_str = asset_details["start_date"] or "2024-11-6"
    print("start_date_str", start_date_str)
    start_date = tz.localize(datetime.strptime(start_date_str, "%Y-%m-%d"))
    print("start_date", start_date)
    model_filename = f"LSTM_model_{asset_id}_{target_column}_{forecast_length}.keras"

    batch_size = 1

    db_url = os.getenv("CONNECTION_STRING")
    db_url_sql = db_url.replace("postgres", "postgresql")
    DATABASE_URL = db_url_sql
    engine = create_engine(DATABASE_URL)

    # Use MetaData to reflect the 'assets_to_forecast' table from the 'forecast' schema
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )
    # Create a new session for database interactions
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    context_length = load_contextlength(SessionLocal, Asset, asset_details)

    def train_and_handle():
        model = train_lstm_model(
            asset_details,
            asset_id,
            df,
            SessionLocal=SessionLocal,
            Asset=Asset,
            tz=tz,
            context_length=context_length,
            forecast_length=forecast_length,
            model_save_path=model_filename,
        )

        print("Length X", len(df))
        save_datalength(SessionLocal, Asset, len(df), asset_details)

    while True:
        end_date = tz.localize(datetime.now())

        if os.path.exists(model_filename):
            print(f"Model {model_filename} exists")
            data_length = load_datalength(SessionLocal, Asset, asset_details)
            print("Data length", data_length)
            df = fetch_pandas_data(
                asset_id,
                start_date,
                end_date,
                target_column,
                asset_details["feature_attributes"],
            )
            if len(df) > data_length * 1.15:
                print("Retraining model")
                train_and_handle()
        else:
            print("Model does not exist")

            df = fetch_pandas_data(
                asset_id,
                start_date,
                end_date,
                target_column,
                asset_details["feature_attributes"],
            )

            print(df.tail(20))
            train_and_handle()

        # Wait for the specified sleep time before running again
        print(f"Sleeping for {sleep_time} seconds before next retraining cycle...")
        time.sleep(sleep_time)
