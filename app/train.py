from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from fetch_and_format_data import format_lstm_data_with_time_features
from keras_tuner.tuners import BayesianOptimization

from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner import HyperParameters
from keras_tuner import Oracle, Tuner
import os


class MyBayesianOptimizationTuner(BayesianOptimization):
    def __init__(
        self,
        *args,
        model_save_path,
        context_length_file_name,
        timestamp_file,
        last_timestep,
        df,
        forecast_length,
        target_column,
        feature_columns,
        batch_size,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_save_path = model_save_path
        self.context_length_file_name = context_length_file_name
        self.timestamp_file = timestamp_file
        self.last_timestep = last_timestep
        self.df = df
        self.forecast_length = forecast_length
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.X = None  # To store training data
        self.y = None  # To store target data

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        print("Checking if the latest trial is the best")

        # Get the current best trial
        best_trial = self.oracle.get_best_trials(num_trials=1)[0]

        # Check if the latest trial is the best trial
        if trial.trial_id == best_trial.trial_id:
            print("Latest trial is the best trial, saving the model and metadata")
            best_model = self.get_best_models(num_models=1)[0]
            best_model.save(self.model_save_path)
            print(f"Model saved to {self.model_save_path}")

            context_length = best_trial.hyperparameters.get("context_length")
            last_timestamp = self.last_timestep

            self.save_metadata(context_length, last_timestamp)

    def save_metadata(self, context_length, last_timestamp):
        with open(self.context_length_file_name, "w") as f:
            f.write(str(context_length))
        with open(self.timestamp_file, "w") as f:
            f.write(str(last_timestamp))
        print(f"Context length saved to {self.context_length_file_name}")
        print(f"Last timestamp saved to {self.timestamp_file}")

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        hp = trial.hyperparameters
        context_length = hp.Int("context_length", min_value=1, max_value=1000)
        X, y, target_timestamps, _, _ = format_lstm_data_with_time_features(
            self.df,
            context_length,
            self.forecast_length,
            self.target_column,
            self.feature_columns,
        )
        self.X = X  # Store data in tuner instance
        self.y = y  # Store target in tuner instance
        self.last_timestep = target_timestamps[-1]

        fit_kwargs["x"] = X
        fit_kwargs["y"] = y
        fit_kwargs["batch_size"] = self.batch_size
        fit_kwargs["validation_split"] = 0.2

        checkpoint_dir = os.path.join(
            "tuner_dir", self.project_name, f"trial_{trial.trial_id}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.weights.h5")

        fit_kwargs["callbacks"] = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint(
                filepath=checkpoint_path, save_best_only=True, save_weights_only=True
            ),
        ]

        model = self.hypermodel.build(hp)
        history = model.fit(
            *fit_args,
            **fit_kwargs,
            epochs=100000,  # Adjust this to the desired number of epochs
        )

        # Save the model explicitly
        model.save(os.path.join(checkpoint_dir, "model.h5"))

        # Extract the final validation loss and return it
        val_loss = history.history["val_loss"][-1]

        # Return the final validation loss in a dictionary
        return {"val_loss": val_loss}

    def load_model(self, trial):
        model = self.hypermodel.build(trial.hyperparameters)
        checkpoint_dir = os.path.join(
            "tuner_dir", self.project_name, f"trial_{trial.trial_id}"
        )
        checkpoint_path = os.path.join(checkpoint_dir, "model.h5")
        model.load_weights(checkpoint_path)
        return model


class LSTMHyperModel(HyperModel):
    def __init__(self, df, forecast_length, target_column, feature_columns, batch_size):
        self.df = df
        self.forecast_length = forecast_length
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.batch_size = batch_size

    def build(self, hp):
        context_length = hp.Int("context_length", min_value=1, max_value=60)

        print("context_length", context_length)
        X, y, target_timestamps, X_last, next_timestamp = (
            format_lstm_data_with_time_features(
                self.df,
                context_length,
                self.forecast_length,
                self.target_column,
                self.feature_columns,
            )
        )
        print("X.shape: ", X.shape)
        print("y.shape: ", y.shape)
        print("x head", X[0])
        print("y head", y[0])
        input_shape = (context_length, X.shape[2])

        model = Sequential()
        model.add(InputLayer(batch_input_shape=(self.batch_size, *input_shape)))

        # Number of LSTM layers
        for i in range(hp.Int("num_lstm_layers", 1, 4)):
            model.add(
                LSTM(
                    units=hp.Int("units", min_value=32, max_value=128, step=32),
                    return_sequences=(i < hp.Int("num_lstm_layers", 1, 4) - 1),
                    stateful=True,
                )
            )
            if hp.Boolean("use_dropout"):
                model.add(
                    Dropout(
                        rate=hp.Float(
                            "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
                        )
                    )
                )

        model.add(Dense(1))  # Dense(1) because we only need one output value

        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice(
                    "learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4]
                )
            ),
            loss="mse",
        )
        return model


def create_lstm_tuned_model(
    df,
    batch_size,
    forecast_length,
    target_column,
    feature_columns,
    project_name="lstm_tuning",
    max_trials=10,
    patience=20,
    model_save_path="best_model.h5",
    context_length_file_name="context_length.txt",
    timestamp_file="timestamp.txt",
):
    df = df[-500:]
    hypermodel = LSTMHyperModel(
        df, forecast_length, target_column, feature_columns, batch_size
    )

    tuner = MyBayesianOptimizationTuner(
        hypermodel,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=1,
        directory="tuner_dir",
        project_name=project_name,
        model_save_path=model_save_path,
        context_length_file_name=context_length_file_name,
        timestamp_file=timestamp_file,
        last_timestep=None,  # Updated later after first trial
        df=df,
        forecast_length=forecast_length,
        target_column=target_column,
        feature_columns=feature_columns,
        batch_size=batch_size,
    )

    tuner.search_space_summary()
    tuner.search()

    best_model = tuner.get_best_models(num_models=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    final_context_length = best_trial.hyperparameters.get("context_length")
    last_timestamp = tuner.last_timestep
    return best_model, tuner.X, last_timestamp, final_context_length


def compile_and_train_model(
    model,
    X_train,
    y_train,
    batch_size,
    epochs=1000000,
    validation_split=0.2,
    patience=20,
):
    learning_rate = float(model.optimizer.learning_rate.numpy())
    print("Learning rate:", learning_rate)
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    # Set up early stopping callback
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
    )

    return model, history
