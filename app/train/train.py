import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from app.get_data.fetch_and_format_data import prepare_data
from app.get_data.api_calls import saveState
import numpy as np


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_save_path, SessionLocal, Asset, asset_details):
        super(CustomCallback, self).__init__()
        self.model_save_path = model_save_path
        self.SessionLocal = SessionLocal
        self.Asset = Asset
        self.asset_details = asset_details
        self.best_val_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # Reset states after each epoch
        for layer in self.model.layers:
            if hasattr(layer, "reset_states") and callable(layer.reset_states):
                layer.reset_states()

        # Check if 'val_loss' improved
        current_val_loss = logs.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            print(
                f"Validation loss improved from {self.best_val_loss} to {current_val_loss}. Saving model."
            )
            self.best_val_loss = current_val_loss
            # Save the model
            self.model.save(self.model_save_path)
            # Call saveState function
            saveState(self.SessionLocal, self.Asset, self.model, self.asset_details)
        else:
            print(f"Validation loss did not improve from {self.best_val_loss}.")

        # TODO: implement saving latest timestamp as well


def build_lstm_model(context_length, num_lstm_layers=2, lstm_units=50, batch_size=1):
    """
    Builds a multi-layer stateful LSTM model for time series forecasting.

    :param context_length: The number of timesteps used for context (input window)
    :param num_lstm_layers: The number of LSTM layers to stack
    :param lstm_units: Number of units in each LSTM layer
    :param batch_size: The size of each batch (needed for stateful LSTMs)
    :return: Compiled LSTM model
    """
    inputs = Input(batch_shape=(batch_size, context_length, 1))

    # Build LSTM layers
    x = LSTM(
        lstm_units,
        activation="tanh",
        stateful=True,
        return_sequences=(num_lstm_layers > 1),
    )(inputs)

    for i in range(1, num_lstm_layers - 1):
        x = LSTM(lstm_units, activation="tanh", stateful=True, return_sequences=True)(x)

    if num_lstm_layers > 1:
        x = LSTM(lstm_units, activation="tanh", stateful=True, return_sequences=False)(
            x
        )

    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss="mse",
    )

    return model


def train_lstm_model(
    asset_details,
    asset_id,
    data,
    SessionLocal,
    Asset,
    context_length,
    forecast_length,
    model_save_path,
    epochs=10,
    validation_split=0.2,
    patience=3,
    batch_size=1,
):
    # Prepare data
    X, y, scaler, last_timestamp = prepare_data(
        data, context_length, forecast_length, asset_details["target_attribute"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, shuffle=False
    )

    # Build model
    model = build_lstm_model(
        context_length, num_lstm_layers=2, lstm_units=50, batch_size=batch_size
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    custom_callback = CustomCallback(
        model_save_path=model_save_path,
        SessionLocal=SessionLocal,
        Asset=Asset,
        asset_details=asset_details,
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, custom_callback],
        shuffle=False,
    )

    # Return the best model and other details
    return model, scaler, last_timestamp
