import pandas as pd


def merge_csv_files(file_paths, output_file):
    """
    Merge multiple CSV files into one based on the 'Date' column and drop rows with any missing values.

    Parameters:
    file_paths (list): List of paths to the CSV files to merge.
    output_file (str): Path to the output CSV file.
    """
    # Initialize an empty DataFrame for merging
    merged_df = pd.DataFrame()

    for file_path in file_paths:
        # Read the current CSV file
        df = pd.read_csv(file_path)

        # Check if the Date column exists
        if "Date" not in df.columns:
            raise ValueError(f"'Date' column not found in {file_path}")

        # Merge with the existing DataFrame
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="Date", how="outer")

    # Drop rows with any missing values
    merged_df.dropna(inplace=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Example usage
    file_paths = [
        "FormattedHistoricalPrices.csv",
        "model_smart_dumb_spread.csv",
        "model_percent_bearish.csv",
    ]
    output_file = "merged_output.csv"

    merge_csv_files(file_paths, output_file)
