import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from app.get_data.fetch_and_format_data import prepare_data
from app.get_data.api_calls import saveState, save_latest_timestamp, save_scaler
import numpy as np


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, model_save_path, SessionLocal, Asset, asset_details, tz, latest_timestamp
    ):
        super(CustomCallback, self).__init__()
        self.model_save_path = model_save_path
        self.SessionLocal = SessionLocal
        self.Asset = Asset
        self.asset_details = asset_details
        self.best_val_loss = np.Inf
        self.latest_timestamp = latest_timestamp
        self.tz = tz

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
            save_latest_timestamp(
                self.SessionLocal,
                self.Asset,
                self.latest_timestamp,
                self.tz,
                self.asset_details,
            )

            saveState(self.SessionLocal, self.Asset, self.model, self.asset_details)
        else:
            print(f"Validation loss did not improve from {self.best_val_loss}.")


def build_lstm_model(
    context_length, num_features, parameters  # Catch-all for additional parameters
):
    """
    Builds a customizable multi-layer stateful LSTM model for time series forecasting.

    :param context_length: Number of timesteps used for context (input window)
    :param num_features: Number of input features
    :param parameters: Dictionary of additional hyperparameters
    :return: Compiled LSTM model
    """
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Dropout, BatchNormalization, Bidirectional

    # Set default values
    num_lstm_layers = parameters.get("num_lstm_layers") or 2
    lstm_units = parameters.get("lstm_units") or 50

    activation = parameters.get("activation") or "tanh"
    learning_rate = parameters.get("learning_rate") or 0.001
    optimizer_type = parameters.get("optimizer_type") or "adam"
    clipnorm = parameters.get("clipnorm")
    loss = parameters.get("loss") or "mean_squared_error"
    dropout_rate = parameters.get("dropout_rate") or 0.0
    recurrent_dropout_rate = parameters.get("recurrent_dropout_rate") or 0.0
    num_dense_layers = parameters.get("num_dense_layers") or 0
    dense_units = parameters.get("dense_units") or 50
    dense_activation = parameters.get("dense_activation") or "relu"
    use_batch_norm = parameters.get("use_batch_norm") or False
    bidirectional = parameters.get("bidirectional") or False
    regularization = parameters.get("regularization") or None
    metrics = parameters.get("metrics") or ["mse"]
    batch_size = 1
    print("all parameters:")
    print("num_lstm_layers", num_lstm_layers)
    print("lstm_units", lstm_units)
    print("activation", activation)
    print("learning_rate", learning_rate)
    print("optimizer_type", optimizer_type)
    print("clipnorm", clipnorm)
    print("loss", loss)
    print("dropout_rate", dropout_rate)
    print("recurrent_dropout_rate", recurrent_dropout_rate)
    print("num_dense_layers", num_dense_layers)
    print("dense_units", dense_units)
    print("dense_activation", dense_activation)
    print("use_batch_norm", use_batch_norm)
    print("bidirectional", bidirectional)
    print("regularization", regularization)
    print("metrics", metrics)
    print("batch_size", batch_size)

    inputs = Input(batch_shape=(batch_size, context_length, num_features))
    x = inputs

    for layer_num in range(num_lstm_layers):
        return_seq = True if layer_num < num_lstm_layers - 1 else False
        lstm_layer = LSTM(
            lstm_units,
            activation=activation,
            stateful=True,
            return_sequences=return_seq,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
        )

        if bidirectional:
            x = Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)

        if use_batch_norm:
            x = BatchNormalization()(x)

    # Add Dense layers
    for dense_layer_num in range(num_dense_layers):
        if regularization and "type" in regularization and "value" in regularization:
            reg = regularizers.get(regularization["type"], None)(
                regularization["value"]
            )
        else:
            reg = None

        x = Dense(
            dense_units,
            activation=dense_activation,
            kernel_regularizer=reg,
        )(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    # Select optimizer
    optimizer_mapping = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adagrad": tf.keras.optimizers.Adagrad,
        "adadelta": tf.keras.optimizers.Adadelta,
        "adamax": tf.keras.optimizers.Adamax,
        "nadam": tf.keras.optimizers.Nadam,
    }

    optimizer_class = optimizer_mapping.get(
        optimizer_type.lower(), tf.keras.optimizers.Adam
    )
    optimizer_kwargs = {"learning_rate": learning_rate}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm

    optimizer = optimizer_class(**optimizer_kwargs)
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics if metrics else [])

    return model


def train_lstm_model(
    asset_details,
    asset_id,
    data,
    SessionLocal,
    Asset,
    tz,
    context_length,
    forecast_length,
    model_save_path,
):
    batch_size = 1
    parameters = asset_details["parameters"] or {}

    epochs = parameters.get("epochs") or 50
    patience = parameters.get("patience") or 5
    validation_split = parameters.get("validation_split") or 0.2
    num_lstm_layers = parameters.get("num_lstm_layers") or 2
    lstm_units = parameters.get("lstm_units") or 50
    # Prepare data
    print(f"Training Parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Patience: {patience}")
    print(f"  Validation Split: {validation_split}")
    X, y, scaler, last_timestamp = prepare_data(
        data,
        context_length,
        forecast_length,
        asset_details["target_attribute"],
        asset_details["feature_attributes"],
    )
    save_scaler(SessionLocal, Asset, scaler, asset_details)
    print("X tail training", X[-3:])
    print("y tail training", y[-3:])
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, shuffle=False
    )
    num_features = X.shape[2]
    # Build model
    model = build_lstm_model(
        context_length,
        num_features,
        parameters,
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
        tz=tz,
        latest_timestamp=last_timestamp,
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
    return model
