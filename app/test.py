# you may add imports only inside the function
# don't change or delete the function definition
def UserFunction(id, eliona):
    from datetime import datetime
    import json

    def write_into_trend(asset_id, timestamp, data, name, prediction_length):
        # Generate the forecast name suffix
        forecast_name_suffix = f"{name}_forecast_{prediction_length}"
        data_dict = {f"{forecast_name_suffix}": float(data)}
        print(data_dict, flush=True)

        # Create the SQL query to insert data into the trend table
        query = f"""
        INSERT INTO trend (ts, asset_id, subtype, data)
        VALUES ('{timestamp}', {asset_id}, 'output', '{json.dumps(data_dict)}')
        """

        # Execute the SQL query
        eliona.SQLQuery(query)
        print(f"Data written to trend: {data_dict}", flush=True)

    # Example usage within UserFunction
    example_asset_id = 10103
    example_timestamp = datetime(2024, 7, 17).isoformat()
    example_data = 75.0
    example_name = "energy"
    example_prediction_length = 24

    # Using write_into_trend to write data into the trend table
    write_into_trend(
        example_asset_id,
        example_timestamp,
        example_data,
        example_name,
        example_prediction_length,
    )
