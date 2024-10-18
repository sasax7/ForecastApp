import json
import pandas as pd


def json_to_csv(json_path, output_csv):
    """
    Converts JSON data from a file to a CSV file.

    Parameters:
    json_path (str): Path to the input JSON file.
    output_csv (str): Path to the output CSV file.
    """
    # Load JSON data from the file
    with open(json_path, "r") as file:
        json_data = json.load(file)

    # Convert the JSON data to a pandas DataFrame
    df = pd.DataFrame(
        json_data["data"],
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "short_lots",
            "long_shorts",
            "positions_long",
            "positions_short",
        ],
    )

    # Replace 'undefined' with NaN
    df.replace("undefined", pd.NA, inplace=True)

    # Drop rows with any NaN values
    df.dropna(inplace=True)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)

    print(f"Data has been saved to {output_csv}")


# Example usage
json_path = "sentiment_EURUSD_DAily.json"  # Path to your JSON file
output_csv = "sentiment_EURUSD_DAily.csv"  # Desired output CSV file path
json_to_csv(json_path, output_csv)
