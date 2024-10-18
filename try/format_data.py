import pandas as pd


def process_historical_prices(
    input_csv, output_csv, ma_periods, ema_periods, macd_params, rsi_period
):

    # Read the CSV data from the file with explicit date format
    df = pd.read_csv(input_csv, parse_dates=["Date"])

    # Strip any leading or trailing spaces in column names
    df.columns = df.columns.str.strip()

    # Print the columns to check if '   ' exists after stripping spaces
    print("Columns in the DataFrame:", df.columns)

    # Check if the required columns are present in the DataFrame
    required_columns = ["Close", "High", "Low", "Open"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"The '{col}' column is missing from the input CSV file.")

    # Sort the DataFrame by date
    df.sort_values("Date", inplace=True)

    # Add time features
    df["Day_of_Week"] = df["Date"].dt.dayofweek  # Monday=0, Sunday=6
    df["Day_of_Month"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month

    # Calculate the differences as percentage changes
    df["Close_Diff"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    df["High_Diff"] = (df["High"] - df["High"].shift(1)) / df["High"].shift(1) * 100
    df["Low_Diff"] = (df["Low"] - df["Low"].shift(1)) / df["Low"].shift(1) * 100
    df["Open_Diff"] = (df["Open"] - df["Open"].shift(1)) / df["Open"].shift(1) * 100

    # Calculate simple moving averages and their percentage differences
    for period in ma_periods:
        ma_column = f"MA_{period}"
        ma_diff_column = f"MA_{period}_Diff"
        df[ma_column] = df["Close"].rolling(window=period).mean()
        df[ma_diff_column] = (df["Close"] - df[ma_column]) / df["Close"] * 100

    # Calculate exponential moving averages and their percentage differences
    for period in ema_periods:
        ema_column = f"EMA_{period}"
        ema_diff_column = f"EMA_{period}_Diff"
        df[ema_column] = df["Close"].ewm(span=period, adjust=False).mean()
        df[ema_diff_column] = (df["Close"] - df[ema_column]) / df["Close"] * 100

    # Calculate MACD and Signal line
    short_ema = df["Close"].ewm(span=macd_params["short_period"], adjust=False).mean()
    long_ema = df["Close"].ewm(span=macd_params["long_period"], adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["MACD_Signal"] = (
        df["MACD"].ewm(span=macd_params["signal_period"], adjust=False).mean()
    )

    # Calculate RSI
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Select the columns to include in the output
    columns_to_include = [
        "Date",
        "Close_Diff",
        "High_Diff",
        "Low_Diff",
        "Open_Diff",
        "Day_of_Week",
        "Day_of_Month",
        "Month",
    ]
    columns_to_include += [f"MA_{period}_Diff" for period in ma_periods]
    columns_to_include += [f"EMA_{period}_Diff" for period in ema_periods]
    columns_to_include += ["MACD", "MACD_Signal", "RSI"]

    difference_df = df[columns_to_include]

    # Drop rows where moving averages are NaN (because of the rolling window)
    difference_df.dropna(inplace=True)

    # Round the differences to three decimal places
    difference_df = difference_df.round(3)

    # Convert 'Date' column to datetime and localize to UTC, then convert to the desired timezone
    difference_df["timestamp"] = pd.to_datetime(difference_df["Date"])
    difference_df.drop(columns=["Date"], inplace=True)

    # Reset index for better readability
    difference_df.reset_index(drop=True, inplace=True)

    # Save the resulting DataFrame to a new CSV file
    difference_df.to_csv(output_csv, index=False)

    # Display the resulting DataFrame
    print(difference_df)
    return difference_df


# Example usage
ma_periods = [50, 200]  # Example simple moving average periods
ema_periods = [9, 21, 50, 100]  # Example exponential moving average periods
macd_params = {
    "short_period": 12,
    "long_period": 26,
    "signal_period": 9,
}  # MACD parameters
rsi_period = 14  # Example RSI period

process_historical_prices(
    "HistoricalPrices.csv",
    "FormattedHistoricalPrices.csv",
    ma_periods,
    ema_periods,
    macd_params,
    rsi_period,
)
