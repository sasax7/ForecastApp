from app.get_data.api_calls import saveState, save_latest_timestamp
import tensorflow as tf
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
        for layer in self.model.layers:
            if hasattr(layer, "reset_states") and callable(layer.reset_states):
                layer.reset_states()


class HyperModelCheckpointCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Reset states of stateful layers
        for layer in self.model.layers:
            if hasattr(layer, "reset_states") and callable(layer.reset_states):
                layer.reset_states()
