import time
import logging
from multiprocessing import Process
import sys
import os
from api.api_calls import update_asset, create_asset

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.api_calls import get_all_assets
from app.data_to_eliona.add_forecast_attributes import (
    add_forecast_attributes_to_all_assets,
)
from app.forecast.forecast import forecast
from app.train.train_and_retrain import train_and_retrain


def background_worker(SessionLocal, Asset):
    create_asset(
        SessionLocal,
        Asset,
        "K01_WP01-f957bc0acd6b3994",
        "Aussentemperatur",
        5,
    )
    with SessionLocal() as session:
        # Fetch all assets marked as 'new'
        new_assets = session.execute(Asset.select()).fetchall()

        # Convert each row to a dictionary
        new_assets_dict = [dict(row._mapping) for row in new_assets]
    all_assets_with_asset_id = add_forecast_attributes_to_all_assets(new_assets_dict)

    for asset_id, asset_details in all_assets_with_asset_id:
        update_asset(
            SessionLocal,
            Asset,
            1,
            processing_status="new",
        )
    while True:
        with SessionLocal() as session:
            # Fetch all assets marked as 'new'
            new_assets = session.execute(
                Asset.select().where(Asset.c.processing_status == "new")
            ).fetchall()

            # Convert each row to a dictionary
            new_assets_dict = [dict(row._mapping) for row in new_assets]

            if not new_assets_dict:
                logging.info("No new assets to process.")
                time.sleep(60)
                continue

            all_assets_with_asset_id = add_forecast_attributes_to_all_assets(
                new_assets_dict
            )

            for asset_id, asset_details in all_assets_with_asset_id:
                logging.info(f"Asset ID: {asset_id}")
                logging.info(f"Asset details: {asset_details}")

                id = asset_details["id"]
                sleep_time_forecast = 60  # 1 minute
                sleep_time_train = 86400  # 24 hours

                # Update the asset's status to 'processing'
                session.execute(
                    Asset.update()
                    .where(Asset.c.id == id)
                    .values(processing_status="processing")
                )
                session.commit()

                # Start forecast and training processes
                forecast_process = Process(
                    target=forecast,
                    args=(
                        asset_details,
                        asset_id,
                        sleep_time_forecast,
                    ),
                )

                train_process = Process(
                    target=train_and_retrain,
                    args=(
                        asset_details,
                        asset_id,
                        sleep_time_train,
                    ),
                )

                # Start both processes
                forecast_process.start()
                # Uncomment if you also want to start the training process simultaneously
                train_process.start()

                logging.info(
                    f"Started forecast and train_and_retrain for asset ID {asset_id}"
                )

        # Sleep before checking again for new assets
        time.sleep(60)
