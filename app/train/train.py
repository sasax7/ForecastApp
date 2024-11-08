import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from app.get_data.fetch_and_format_data import prepare_data
from app.get_data.api_calls import saveState


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
    context_length,
    forecast_length,
    model_save_path,
    epochs=10,
    validation_split=0.2,
    patience=3,
    batch_size=1,
):
    """
    Trains a stateful LSTM model and returns the trained model and scaler.

    :param asset_id: Identifier for the asset (used for saving models/scalers)
    :param data: Pandas DataFrame containing the 'timestamp' and 'brightness' columns
    :param context_length: The number of timesteps used for context (input window)
    :param forecast_length: The number of timesteps ahead to predict (not used here since we're predicting one value)
    :param model_save_path: Path to save the trained model
    :param epochs: Number of training epochs
    :param validation_split: Fraction of data to use for validation
    :param patience: Patience for early stopping
    :param batch_size: Batch size for training (needed for stateful LSTMs)
    :return: Trained model and scaler
    """
    X, y, scaler = prepare_data(
        data, context_length, forecast_length, asset_details["target_attribute"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, shuffle=False
    )

    model = build_lstm_model(
        context_length, num_lstm_layers=2, lstm_units=50, batch_size=batch_size
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            shuffle=False,
        )

        saveState(
            model,
            f"{asset_id}_{asset_details['target_attribute']}_{asset_details['forecast_length']}_lstm_states.pkl",
        )
        for layer in model.layers:
            if hasattr(layer, "reset_states") and callable(layer.reset_states):
                layer.reset_states()

    model.save(model_save_path)

    return model, scaler
