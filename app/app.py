from api.api_calls import (
    create_asset,
    update_asset,
    get_asset_by_id,
    delete_asset,
    get_all_assets,
)
import sys
import os
import subprocess

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import the module
from add_forecast_attributes import add_forecast_attributes_to_all_assets


def start_forecast(SessionLocal, Asset):
    # create_asset(SessionLocal, Asset, "Environment sensor room 1", "brightness", 3)

    all_assets = get_all_assets(SessionLocal, Asset)
    all_assets_with_asset_id = add_forecast_attributes_to_all_assets(all_assets)
    print("All assets with asset ID:", all_assets_with_asset_id)

    for asset_id, asset_details in all_assets_with_asset_id:
        print(f"Asset ID: {asset_id}")
        print(f"Asset Details: {asset_details}")

        forecast_process = subprocess.Popen(
            [
                "forecast/forecast.py",
                "--asset_id",
                str(asset_id),
                "--forecast_length",
                str(asset_details["forecast_length"]),
                "--target_column",
                asset_details["target_attribute"],
                "--feature_columns",
                asset_details["feature_attributes"],
            ]
        )
        print(f"Started forecast.py for asset ID {asset_id}")

        # Start `train_and_retrain.py` script with correct arguments
        train_process = subprocess.Popen(
            [
                "forecast/train_and_retrain.py",
                "--asset_id",
                str(asset_id),
                "--start_date",
                asset_details["start_date"] or "2020-1-1",  # Use the correct start date
                "--forecast_length",
                str(asset_details["forecast_length"]),
                "--target_column",
                asset_details["target_attribute"],
                "--feature_columns",
                asset_details["feature_attributes"],
            ]
        )
