from api.api_calls import (
    create_asset,
    update_asset,
    get_asset_by_id,
    delete_asset,
    get_all_assets,
)
import sys
import os
import multiprocessing


# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from add_forecast_attributes import add_forecast_attributes_to_all_assets
from forecast import forecast
from train_and_retrain import train_and_retrain


def start_forecast_and_train(SessionLocal, Asset):
    all_assets = get_all_assets(SessionLocal, Asset)
    all_assets_with_asset_id = add_forecast_attributes_to_all_assets(all_assets)

    for asset_id, asset_details in all_assets_with_asset_id:
        print(f"Asset ID: {asset_id}")
        print(f"Asset details: {asset_details}")

        # # Set the sleep time for the retrain cycle (e.g., 24 hours in seconds)
        # sleep_time = 60  # 1 minute
        # # Run forecast and train_and_retrain in parallel
        # forecast_process = multiprocessing.Process(
        #     target=forecast,
        #     args=(
        #         asset_id,
        #         asset_details["forecast_length"],
        #         asset_details["target_attribute"],
        #         asset_details["feature_attributes"],
        #         sleep_time,
        #     ),
        # )
        sleep_time = 86400  # 24 hours

        train_process = multiprocessing.Process(
            target=train_and_retrain,
            args=(
                asset_id,
                asset_details["start_date"] or "2020-1-1",
                asset_details["forecast_length"],
                asset_details["target_attribute"],
                asset_details["feature_attributes"] or "",
                sleep_time,
            ),
        )

        # Start both processes
        # forecast_process.start()
        train_process.start()

        print(f"Started forecast and train_and_retrain for asset ID {asset_id}")
