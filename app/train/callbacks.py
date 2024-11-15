from app.get_data.api_calls import saveState, save_latest_timestamp
import tensorflow as tf
import numpy as np
from kerastuner.tuners import BayesianOptimization
from kerastuner.engine import trial as trial_module
import traceback
import shutil
import os


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

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if tf.math.is_nan(loss):
                print(f"Batch {batch}: Invalid loss, terminating training.")
                self.model.stop_training = True


class MyTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        try:
            return super(MyTuner, self).run_trial(trial, *args, **kwargs)
        except Exception as e:
            print(f"Trial {trial.trial_id} failed due to exception: {e}")
            # Optionally, print the traceback for debugging
            import traceback

            traceback.print_exc()
            # Mark the trial as COMPLETED with val_loss as infinity
            self.oracle.update_trial(
                trial.trial_id,
                trial_status=trial_module.TrialStatus.COMPLETED,
                metrics={"val_loss": float("inf")},
            )
            # End the trial
            self.oracle.end_trial(
                trial_id=trial.trial_id, trial_status=trial_module.TrialStatus.COMPLETED
            )
            # Return a metrics dictionary with val_loss as infinity
            return {"val_loss": float("inf")}
