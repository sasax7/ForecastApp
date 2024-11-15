import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import argparse
import time
from app.data_to_eliona.create_asset_to_save_models import (
    load_model_from_eliona,
    save_model_to_eliona,
    model_exists,
)


def load_series(
    X_filename, y_filename, timestamps_filename, original_close_diff_filename
):
    timestamp = np.load(timestamps_filename)
    original_close_diff = np.load(original_close_diff_filename)
    X = np.load(X_filename)
    y = np.load(y_filename)
    return X, y, timestamp, original_close_diff


def build_model(input_shape, batch_size, learning_rate):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(batch_size, *input_shape)))
    model.add(
        LSTM(
            50,
            stateful=True,
            return_sequences=True,
        )
    )
    model.add(
        LSTM(
            50,
            stateful=True,
            return_sequences=True,
        )
    )
    model.add(LSTM(50, stateful=True))
    model.add(Dense(1))  # Single output unit for regression
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model


def load_data(
    X_filename, y_filename, timestamps_filename, original_close_diff_filename, test_size
):
    X, y, timestamps, original_close_diff = load_series(
        X_filename, y_filename, timestamps_filename, original_close_diff_filename
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    timestamps_train, timestamps_val, close_diff_train, close_diff_val = (
        train_test_split(
            timestamps, original_close_diff, test_size=test_size, shuffle=False
        )
    )
    return (
        X_train,
        X_val,
        y_train,
        y_val,
        timestamps_train,
        timestamps_val,
        close_diff_train,
        close_diff_val,
    )


def safe_save_model(model, filepath, attempts=5, delay=1):
    """Attempt to save the model with retries on failure due to file access issues."""
    for attempt in range(attempts):
        try:
            save_model_to_eliona(model, filepath)
            print(f"Model saved successfully to {filepath}")
            return True
        except PermissionError as e:
            if attempt < attempts - 1:
                print(f"PermissionError: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(
                    f"Failed to save the model to {filepath} after {attempts} attempts."
                )
                raise


def train_and_save_model(
    X_train,
    y_train,
    input_shape,
    batch_size,
    epochs,
    patience,
    learning_rate,
    model_filename,
):
    model = build_model(input_shape, batch_size, learning_rate)

    # Define early stopping based on validation loss
    early_stopping = EarlyStopping(
        patience=patience, restore_best_weights=True, monitor="val_loss", mode="min"
    )

    # Define model checkpoint to save the best model based on validation loss
    model_checkpoint = ModelCheckpoint(
        model_filename,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    # Train the model with early stopping and model checkpoint callbacks
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping, model_checkpoint],
        shuffle=False,
    )

    # Try saving the final model safely
    safe_save_model(model, model_filename)


def evaluate_and_predict_model(
    X_val,
    y_val,
    timestamps_val,
    close_diff_val,
    batch_size,
    buffer,
    model_filename,
):
    model = load_model_from_eliona(model_filename)
    predictions = model.predict(X_val, batch_size=batch_size).flatten()

    total_predictions = len(predictions)
    summ_wins = 0
    summ_losses = 0
    max_win = 0
    max_loss = 0
    num_wins = 0
    num_losses = 0

    for i in range(total_predictions):
        predicted_change = predictions[i]
        true_change = y_val[i]
        timestamps = timestamps_val[i]

        if -buffer < predicted_change < buffer:
            predicted_change = 0
        print(
            f"True change: {true_change} at time: {timestamps}, Predicted change: {predicted_change}"
        )
        if not (predicted_change == 0):
            if predicted_change * true_change > 0:
                win_amount = abs(close_diff_val[i])
                summ_wins += win_amount
                num_wins += 1
                max_win = max(max_win, win_amount)
            else:
                loss_amount = abs(close_diff_val[i])
                summ_losses += loss_amount
                num_losses += 1
                max_loss = max(max_loss, loss_amount)

    total_filtered_predictions = num_wins + num_losses
    profit_amount = summ_wins - summ_losses
    winrate = (
        (num_wins / total_filtered_predictions * 100)
        if total_filtered_predictions > 0
        else 0
    )
    rr_ratio = (summ_wins / summ_losses) if summ_losses > 0 else np.inf
    expectancy_per_trade = (
        (profit_amount / total_filtered_predictions)
        if total_filtered_predictions > 0
        else np.inf
    )
    avg_win = summ_wins / num_wins if num_wins > 0 else 0
    avg_loss = summ_losses / num_losses if num_losses > 0 else 0

    print(
        f"Total predictions: {total_filtered_predictions} of {total_predictions} trades"
    )
    print(f"Total wins amount: {summ_wins}, Total losses amount: {summ_losses}")
    print("------------------------------------")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Total profit made: {profit_amount}")
    print(f"Max win: {max_win}, Max loss: {max_loss}")
    print(f"Risk Reward Ratio: {rr_ratio}")
    print(f"Expectancy per trade: {expectancy_per_trade}")
    print(f"Average win: {avg_win}")
    print(f"Average loss: {avg_loss}")

    return predictions


def main():
    # Initializations
    X_filename = "X_series.npy"
    y_filename = "y_series.npy"
    timestamps_filename = "timestamps.npy"
    original_close_diff_filename = "original_close_diff.npy"
    parser = argparse.ArgumentParser(
        description="Train and evaluate an LSTM model for stock prediction."
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.5,
        help="Buffer value for prediction threshold.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10000, help="Number of epochs for training."
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Patience for early stopping."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        help="Proportion of data to use for validation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate for the optimizer.",
    )
    args = parser.parse_args()

    # Use the parsed arguments
    buffer = args.buffer
    batch_size = 1
    epochs = args.epochs
    patience = args.patience
    test_size = args.test_size
    learning_rate = args.learning_rate
    # best so far: learning_rate = 0.00001, patience = 3, buffer = 0.5

    # Model filename
    model_filename = f"learningrate{learning_rate}_patience{patience}_model.h5"

    # Load data
    (
        X_train,
        X_val,
        y_train,
        y_val,
        timestamps_train,
        timestamps_val,
        close_diff_train,
        close_diff_val,
    ) = load_data(
        X_filename,
        y_filename,
        timestamps_filename,
        original_close_diff_filename,
        test_size,
    )

    # Check if model exists
    if not model_exists(model_filename):
        # Build, train, and save the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        train_and_save_model(
            X_train,
            y_train,
            input_shape,
            batch_size,
            epochs,
            patience,
            learning_rate,
            model_filename,
        )

    # Evaluate and make predictions
    evaluate_and_predict_model(
        X_val,
        y_val,
        timestamps_val,
        close_diff_val,
        batch_size,
        buffer,
        model_filename,
    )


if __name__ == "__main__":
    main()
