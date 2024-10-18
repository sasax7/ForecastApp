import numpy as np
import pandas as pd


def create_series(input_csv, context_length, target_column, forecast_length):
    df = pd.read_csv(input_csv, parse_dates=["timestamp"])

    # Ensure the target column is present
    if target_column not in df.columns:
        raise KeyError(
            f"The target column '{target_column}' is missing from the input CSV file."
        )

    # Create the input features and target values
    X, y, timestamps, original_close_diff = [], [], [], []
    for i in range(len(df) - context_length - forecast_length + 1):
        X.append(df.iloc[i : i + context_length].drop(columns=["timestamp"]).values)
        target_values = df.iloc[
            i + context_length : i + context_length + forecast_length
        ][target_column].values
        y.append(target_values)

        # Capture timestamps and original close difference
        timestamps.append(
            df.iloc[i + context_length : i + context_length + forecast_length][
                "timestamp"
            ].values
        )
        original_close_diff.append(
            df.iloc[i + context_length : i + context_length + forecast_length][
                target_column
            ].values
        )

    return np.array(X), np.array(y), np.array(timestamps), np.array(original_close_diff)


def save_series(X, y, timestamps, original_close_diff, X_filename, y_filename):
    np.save(X_filename, X)
    np.save(y_filename, y)
    np.save("timestamps.npy", timestamps)
    np.save("original_close_diff.npy", original_close_diff)


# Example usage
input_csv = "FormattedHistoricalPrices.csv"
context_length = 30  # Number of past observations
target_column = "Close_Diff"  # Column to predict
forecast_length = 1  # Number of future observations to predict

# Create the series
X, y, timestamps, original_close_diff = create_series(
    input_csv, context_length, target_column, forecast_length
)
print("X.shape:", X.shape)
print("y.shape:", y.shape)
print("timestamps.shape:", timestamps.shape)
print("original_close_diff.shape:", original_close_diff.shape)
print("X head:", X[0])
print("y head:", y[-20:])
print("timestamps head:", timestamps[0])
print("original_close_diff head:", original_close_diff[0])

# Save the series to disk
save_series(X, y, timestamps, original_close_diff, "X_series.npy", "y_series.npy")

# Calculate and print some statistics
y_flat = y.flatten()
num_positive = np.sum(y_flat > 0)
num_negative = np.sum(y_flat < 0)
num_zero = np.sum(y_flat == 0)

print(f"Number of positive changes in 'Close_Diff': {num_positive}")
print(f"Number of negative changes in 'Close_Diff': {num_negative}")
print(f"Number of no changes in 'Close_Diff': {num_zero}")
print(f"Percentage of positive changes in 'Close_Diff': {num_positive / len(y_flat)}")
print(f"Percentage of negative changes in 'Close_Diff': {num_negative / len(y_flat)}")
print(f"Percentage of no changes in 'Close_Diff': {num_zero / len(y_flat)}")
