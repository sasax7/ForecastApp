import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    Dropout,
    BatchNormalization,
    Bidirectional,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    Nadam,
)
from kerastuner import HyperModel, HyperParameters
from kerastuner.tuners import RandomSearch


class LSTMHyperModel(HyperModel):
    def __init__(
        self, context_length, num_features, parameters=None, hyperparameters=None
    ):
        self.context_length = context_length
        self.num_features = num_features
        self.parameters = parameters or {}
        self.hyperparameters = self.validate_hyperparameters(hyperparameters)

    def validate_hyperparameters(self, hyperparameters):
        if not hyperparameters:
            return {}

        validated_hyperparameters = {}

        for key, params in hyperparameters.items():
            if isinstance(params, dict):
                # For hp.Int and hp.Float
                required_keys = {"min_value", "max_value", "step"}
                if required_keys.issubset(params.keys()):
                    validated_hyperparameters[key] = params
                else:
                    print(
                        f"Warning: Hyperparameter '{key}' is missing required keys. Using standard values."
                    )
            elif isinstance(params, list):
                # For hp.Choice
                validated_hyperparameters[key] = params
            elif isinstance(params, bool):
                # For hp.Boolean
                validated_hyperparameters[key] = params
            else:
                print(
                    f"Warning: Hyperparameter '{key}' has an unsupported format. Using standard values."
                )

        return validated_hyperparameters

    def build(self, hp):
        inputs = Input(batch_shape=(1, self.context_length, self.num_features))
        x = inputs

        # Hyperparameters to tune, with initial values from initial_params if provided

        # Activation functions (hp.Choice)
        activation = hp.Choice(
            "activation",
            values=self.hyperparameters.get("activation")
            or ["tanh", "relu", "sigmoid", "linear"],
        )
        dense_activation = hp.Choice(
            "dense_activation",
            values=self.hyperparameters.get("dense_activation")
            or ["tanh", "relu", "sigmoid", "linear"],
        )
        num_lstm_layers = hp.Int(
            "num_lstm_layers",
            min_value=self.hyperparameters.get("num_lstm_layers")
            or {}.get("min_value")
            or 1,
            max_value=self.hyperparameters.get("num_lstm_layers")
            or {}.get("max_value")
            or 4,
            step=self.hyperparameters.get("num_lstm_layers") or {}.get("step") or 1,
        )
        lstm_units = hp.Int(
            "lstm_units",
            min_value=self.hyperparameters.get("lstm_units")
            or {}.get("min_value")
            or 32,
            max_value=self.hyperparameters.get("lstm_units")
            or {}.get("max_value")
            or 256,
            step=self.hyperparameters.get("lstm_units") or {}.get("step") or 32,
        )
        dropout_rate = hp.Float(
            "dropout_rate",
            min_value=self.hyperparameters.get("dropout_rate")
            or {}.get("min_value")
            or 0.0,
            max_value=self.hyperparameters.get("dropout_rate")
            or {}.get("max_value")
            or 0.5,
            step=self.hyperparameters.get("dropout_rate") or {}.get("step") or 0.1,
        )
        recurrent_dropout_rate = hp.Float(
            "recurrent_dropout_rate",
            min_value=self.hyperparameters.get("recurrent_dropout_rate")
            or {}.get("min_value")
            or 0.0,
            max_value=self.hyperparameters.get("recurrent_dropout_rate")
            or {}.get("max_value")
            or 0.5,
            step=self.hyperparameters.get("recurrent_dropout_rate")
            or {}.get("step")
            or 0.1,
        )
        num_dense_layers = hp.Int(
            "num_dense_layers",
            min_value=self.hyperparameters.get("num_dense_layers")
            or {}.get("min_value")
            or 0,
            max_value=self.hyperparameters.get("num_dense_layers")
            or {}.get("max_value")
            or 3,
            step=self.hyperparameters.get("num_dense_layers") or {}.get("step") or 1,
        )
        dense_units = hp.Int(
            "dense_units",
            min_value=self.hyperparameters.get("dense_units")
            or {}.get("min_value")
            or 32,
            max_value=self.hyperparameters.get("dense_units")
            or {}.get("max_value")
            or 256,
            step=self.hyperparameters.get("dense_units") or {}.get("step") or 32,
        )
        learning_rate = hp.Float(
            "learning_rate",
            min_value=self.hyperparameters.get("learning_rate")
            or {}.get("min_value")
            or 1e-4,
            max_value=self.hyperparameters.get("learning_rate")
            or {}.get("max_value")
            or 1e-2,
            sampling="log",
        )
        optimizer_type = hp.Choice(
            "optimizer_type",
            values=["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"],
        )
        use_batch_norm = hp.Boolean(
            "use_batch_norm",
        )

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

            x = lstm_layer(x)

            if use_batch_norm:
                x = BatchNormalization()(x)

        for _ in range(num_dense_layers):
            x = Dense(dense_units, activation=dense_activation)(x)
            if use_batch_norm:
                x = BatchNormalization()(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        outputs = Dense(1)(x)

        model = Model(inputs, outputs)

        optimizer_mapping = {
            "adam": Adam,
            "sgd": SGD,
            "rmsprop": RMSprop,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "nadam": Nadam,
        }

        optimizer_class = optimizer_mapping[optimizer_type]
        optimizer = optimizer_class(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=self.parameters.get("loss", "mean_squared_error"),
        )

        return model
