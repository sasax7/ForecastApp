from config import apps_api


def Initialize():

    app = apps_api.get_app_by_name("forecast")

    if not app.registered:
        apps_api.patch_app_by_name("forecast", True)
        print("App 'forecast' registered.")

    # else:
    #     print("App 'forecast' already active.")

    # # Set up the database session and assets table
    # SessionLocal, Asset = setup_database(db_url_sql)

    # # Create a new session
    # session = SessionLocal()

    # try:

    #     assets = get_all_assets(session, Asset)
    #     print("All assets:", assets)

    #     # Loop through each asset and apply forecast attributes
    #     for asset in assets:
    #         asset_id = asset[0]  # Assuming the first element is asset_id
    #         asset_type = asset[1]  # Assuming the second element is asset_type
    #         forecast_length = asset[2]  # Assuming the third element is forecast_length

    #         # Create the forecast_name_suffix based on the asset_type and forecast_length
    #         forecast_name_suffix = f"{asset_type}_forecast_{forecast_length}"

    #         # The attributes to forecast would be a list, in this case using the asset_type as an example
    #         attributes_to_forecast = [asset_type]

    #         # Call add_forecast_attributes with the asset details
    #         add_forecast_attributes(
    #             asset_id, attributes_to_forecast, forecast_name_suffix
    #         )
    #         print(f"Forecast attributes added for asset ID {asset_id}")

    #         # Start `forecast.py` script with correct arguments
    #         forecast_process = subprocess.Popen(
    #             [
    #                 python_executable,
    #                 "forecast/forecast.py",
    #                 "--asset_id",
    #                 str(asset_id),
    #                 "--forecast_length",
    #                 str(forecast_length),
    #                 "--target_column",
    #                 asset_type,
    #                 "--feature_columns",
    #                 asset_type,
    #             ]
    #         )
    #         print(f"Started forecast.py for asset ID {asset_id}")

    #         # Start `train_and_retrain.py` script with correct arguments
    #         train_process = subprocess.Popen(
    #             [
    #                 python_executable,
    #                 "forecast/train_and_retrain.py",
    #                 "--asset_id",
    #                 str(asset_id),
    #                 "--start_date",
    #                 "2020-1-1",  # Use the correct start date
    #                 "--forecast_length",
    #                 str(forecast_length),
    #                 "--target_column",
    #                 asset_type,
    #                 "--feature_columns",
    #                 asset_type,
    #             ]
    #         )
    #         print(f"Started train_and_retrain.py for asset ID {asset_id}")

    # finally:
    #     # Close the session when done
    #     session.close()
