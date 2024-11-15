import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from app.get_data.fetch_and_format_data import prepare_data
from app.get_data.api_calls import (
    saveState,
    save_latest_timestamp,
    save_scaler,
    save_parameters,
)

import numpy as np
from app.train.lstmHyperModel import LSTMHyperModel
from kerastuner.tuners import BayesianOptimization
from app.train.build_standard_lstm import build_lstm_model
from app.train.callbacks import CustomCallback, HyperModelCheckpointCallback, MyTuner


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

    trainingparameters = asset_details["trainingparameters"] or {}
    epochs = trainingparameters.get("epochs", 50)
    patience = trainingparameters.get("patience", 5)
    validation_split = trainingparameters.get("validation_split", 0.2)
    objective = trainingparameters.get("objective", "val_loss")

    hyperparameters = asset_details["hyperparameters"] or {}
    hyperparameters_percent_data = hyperparameters.get("percent_data", 0.01)
    max_trials = hyperparameters.get("max_trials", 100)
    project_name = f"hyperparameters_model_{asset_id}_{asset_details['target_attribute']}_{forecast_length}"

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

    # Split the data
    num_features = X.shape[2]
    if hyperparameters or ((not hyperparameters) and (not parameters)):
        data_length = int(len(X) * hyperparameters_percent_data)
        X_hyper = X[:data_length]
        y_hyper = y[:data_length]

        X_train_hyper, X_val_hyper, y_train_hyper, y_val_hyper = train_test_split(
            X_hyper, y_hyper, test_size=validation_split, shuffle=False
        )
        # Hyperparameter optimization
        hypermodel = LSTMHyperModel(
            context_length,
            num_features,
            parameters=parameters,
            hyperparameters=hyperparameters,
        )
        tuner = MyTuner(
            hypermodel,
            objective=objective,
            max_trials=max_trials,
            directory="hyperparameter_search",
            project_name=project_name,
        )

        tuner.search_space_summary()
        hypermodel_checkpoint_callback = HyperModelCheckpointCallback()
        tuner.search(
            X_train_hyper,
            y_train_hyper,
            epochs=epochs,
            batch_size=1,
            validation_data=(X_val_hyper, y_val_hyper),
            callbacks=[
                EarlyStopping(
                    monitor=objective,
                    patience=parameters.get("patience") or 5,
                    restore_best_weights=True,
                ),
                hypermodel_checkpoint_callback,
            ],
        )

        tuner.results_summary()

        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:")
        print(best_hyperparameters)
        print("Best hyperparameters values:")
        print(best_hyperparameters.values)
        # Build the model with the best hyperparameters

        model = build_lstm_model(
            context_length,
            num_features,
            best_hyperparameters.values,
        )

        # Train the model with the full dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor=objective, patience=patience, restore_best_weights=True
        )
        custom_callback = CustomCallback(
            model_save_path=model_save_path,
            SessionLocal=SessionLocal,
            Asset=Asset,
            asset_details=asset_details,
            tz=tz,
            latest_timestamp=last_timestamp,
        )
        save_parameters(SessionLocal, Asset, best_hyperparameters.values, asset_details)
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

        return model

    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        model = build_lstm_model(
            context_length,
            num_features,
            parameters,
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor=objective, patience=patience, restore_best_weights=True
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
