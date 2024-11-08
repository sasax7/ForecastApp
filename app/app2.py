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

from app.data_to_eliona.add_forecast_attributes import (
    add_forecast_attributes_to_all_assets,
)
from app.forecast.forecast import forecast
from app.train.train_and_retrain import train_and_retrain


def start_forecast_and_train(SessionLocal, Asset):
    all_assets = get_all_assets(SessionLocal, Asset)
    all_assets_with_asset_id = add_forecast_attributes_to_all_assets(all_assets)
    create_asset(SessionLocal, Asset, "Environment sensor room 1", "brightness", 5)
    for asset_id, asset_details in all_assets_with_asset_id:
        print(f"Asset ID: {asset_id}")
        print(f"Asset details: {asset_details}")
        id = asset_details["id"]
        sleep_time_forecast = 60  # 1 minute
        sleep_time_train = 86400  # 24 hours

        forecast_process = multiprocessing.Process(
            target=forecast,
            args=(
                asset_details,
                id,
                asset_id,
                asset_details["forecast_length"],
                asset_details["target_attribute"],
                asset_details.get("feature_attributes", ""),
                sleep_time_forecast,
            ),
        )

        train_process = multiprocessing.Process(
            target=train_and_retrain,
            args=(
                asset_details,
                asset_id,
                id,
                asset_details.get("start_date") or "2020-1-1",
                asset_details["forecast_length"],
                asset_details["target_attribute"],
                asset_details.get("feature_attributes", ""),
                sleep_time_train,
            ),
        )

        # Start both processes
        train_process.start()
        # forecast_process.start()

        print(f"Started forecast and train_and_retrain for asset ID {asset_id}")
