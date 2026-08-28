from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Workload


class TensorFlowWorkload(Workload):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow backend requested but tensorflow is not installed"
            ) from exc

        self.tf = tf
        self.architecture = config["ai"]["architecture"]
        self.variant = config["ai"]["variant"]
        self.application = config["ai"]["application"]
        self.num_classes = int(config["ai"]["num_classes"])
        self.model = self._build_model()
        self.learning_rate = float(
            config["execution"].get("learning_rate", 1e-3)
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

    def _build_model(self):
        tf = self.tf
        arch = self.architecture
        variant = self.variant
        input_size = int(self.config["ai"]["input_size"])

        if arch == "resnet":
            builders = {
                "resnet50": tf.keras.applications.ResNet50,
                "resnet101": tf.keras.applications.ResNet101,
            }
            builder = builders.get(variant)
            if builder is None:
                raise ValueError(
                    f"Native TensorFlow builder not implemented for "
                    f"architecture={arch!r}, variant={variant!r}"
                )
            return builder(
                weights=None,
                input_shape=(input_size, input_size, 3),
                classes=self.num_classes,
            )

        if arch == "mobilenet":
            builders = {
                "mobilenet_v2": tf.keras.applications.MobileNetV2,
                "mobilenet_v3_small": tf.keras.applications.MobileNetV3Small,
                "mobilenet_v3_large": tf.keras.applications.MobileNetV3Large,
            }
            builder = builders.get(variant)
            if builder is None:
                raise ValueError(
                    f"Native TensorFlow builder not implemented for "
                    f"architecture={arch!r}, variant={variant!r}"
                )
            return builder(
                weights=None,
                input_shape=(input_size, input_size, 3),
                classes=self.num_classes,
            )

        if arch == "efficientnet":
            builders = {
                "efficientnet_b0": tf.keras.applications.EfficientNetB0,
                "efficientnet_b1": tf.keras.applications.EfficientNetB1,
                "efficientnet_b2": tf.keras.applications.EfficientNetB2,
            }
            builder = builders.get(variant)
            if builder is None:
                raise ValueError(
                    f"Native TensorFlow builder not implemented for "
                    f"architecture={arch!r}, variant={variant!r}"
                )
            return builder(
                weights=None,
                input_shape=(input_size, input_size, 3),
                classes=self.num_classes,
            )

        if arch in {"lstm", "gru"}:
            layers, bidirectional = _rnn_variant_settings(
                architecture=arch,
                variant=variant,
            )
            if self.application == "text_classification":
                return self._build_text_rnn(
                    kind=arch,
                    num_layers=layers,
                    bidirectional=bidirectional,
                )
            return self._build_sequence_rnn(
                kind=arch,
                num_layers=layers,
                bidirectional=bidirectional,
            )

        if arch == "tiny_transformer":
            layer_map = {
                "tiny_transformer_2layer": 2,
                "tiny_transformer_4layer": 4,
                "tiny_transformer_6layer": 6,
            }
            layers = layer_map.get(variant)
            if layers is None:
                raise ValueError(
                    f"Unsupported native tiny transformer variant: {variant!r}"
                )
            return self._build_tiny_transformer(layers)

        if arch == "dense_autoencoder":
            depth_map = {
                "dense_autoencoder_2layer": 2,
                "dense_autoencoder_3layer": 3,
                "dense_autoencoder_5layer": 5,
            }
            depth = depth_map.get(variant)
            if depth is None:
                raise ValueError(f"Unsupported dense autoencoder: {variant!r}")
            return self._build_dense_autoencoder(input_size, depth)

        if arch == "convolutional_autoencoder":
            depth_map = {
                "convolutional_autoencoder_2layer": 2,
                "convolutional_autoencoder_4layer": 4,
                "convolutional_autoencoder_6layer": 6,
            }
            depth = depth_map.get(variant)
            if depth is None:
                raise ValueError(
                    f"Unsupported convolutional autoencoder: {variant!r}"
                )
            return self._build_conv_autoencoder(input_size, depth)

        if arch == "variational_autoencoder":
            if variant == "vae_fc":
                return self._build_fc_vae(input_size, beta=1.0)
            if variant in {"vae_conv", "beta_vae"}:
                return self._build_conv_vae(
                    input_size,
                    beta=4.0 if variant == "beta_vae" else 1.0,
                )
            raise ValueError(f"Unsupported VAE variant: {variant!r}")

        if arch == "feedforward_mlp":
            depth_map = {
                "mlp_2layer": 2,
                "mlp_4layer": 4,
                "mlp_8layer": 8,
            }
            depth = depth_map.get(variant)
            if depth is None:
                raise ValueError(f"Unsupported MLP variant: {variant!r}")
            return self._build_mlp(depth)

        raise ValueError(
            f"Native TensorFlow builder not implemented for "
            f"architecture={arch!r}, variant={variant!r}. "
            "Use a supported native variant or an artifact-backed runtime."
        )

    def _rnn_layer(
        self,
        layer_cls,
        return_sequences: bool,
        bidirectional: bool,
    ):
        layer = layer_cls(128, return_sequences=return_sequences)
        if bidirectional:
            return self.tf.keras.layers.Bidirectional(layer)
        return layer

    def _build_sequence_rnn(
        self,
        kind: str,
        num_layers: int,
        bidirectional: bool,
    ):
        tf = self.tf
        seq = int(self.config["ai"]["sequence_length"])
        dim = int(self.config["ai"]["input_dim"])
        inputs = tf.keras.Input(shape=(seq, dim))
        layer_cls = tf.keras.layers.LSTM if kind == "lstm" else tf.keras.layers.GRU

        x = inputs
        for index in range(num_layers):
            x = self._rnn_layer(
                layer_cls=layer_cls,
                return_sequences=index < num_layers - 1,
                bidirectional=bidirectional,
            )(x)

        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        return tf.keras.Model(inputs, outputs)

    def _build_text_rnn(
        self,
        kind: str,
        num_layers: int,
        bidirectional: bool,
    ):
        tf = self.tf
        length = int(self.config["ai"]["max_text_length"])
        vocab = int(self.config["ai"]["vocab_size"])
        inputs = tf.keras.Input(shape=(length,), dtype="int32")
        x = tf.keras.layers.Embedding(vocab, 128)(inputs)
        layer_cls = tf.keras.layers.LSTM if kind == "lstm" else tf.keras.layers.GRU

        for index in range(num_layers):
            x = self._rnn_layer(
                layer_cls=layer_cls,
                return_sequences=index < num_layers - 1,
                bidirectional=bidirectional,
            )(x)

        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        return tf.keras.Model(inputs, outputs)

    def _build_tiny_transformer(self, layer_count: int):
        tf = self.tf
        length = int(self.config["ai"]["max_text_length"])
        vocab = int(self.config["ai"]["vocab_size"])
        embed = 128

        inputs = tf.keras.Input(shape=(length,), dtype="int32")
        tokens = tf.keras.layers.Embedding(vocab, embed)(inputs)
        positions = tf.range(start=0, limit=length, delta=1)
        pos_embedding = tf.keras.layers.Embedding(length, embed)(positions)
        x = tokens + pos_embedding

        for _ in range(layer_count):
            attn = tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=32,
            )(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attn)
            ff = tf.keras.layers.Dense(256, activation="relu")(x)
            ff = tf.keras.layers.Dense(embed)(ff)
            x = tf.keras.layers.LayerNormalization()(x + ff)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        return tf.keras.Model(inputs, outputs)

    def _build_dense_autoencoder(self, input_size: int, depth: int):
        tf = self.tf
        flat = 3 * input_size * input_size
        if depth == 2:
            dims = [256, flat]
        elif depth == 3:
            dims = [512, 128, flat]
        elif depth == 5:
            dims = [512, 256, 64, 256, flat]
        else:
            raise ValueError(f"Unsupported dense autoencoder depth: {depth}")
        inputs = tf.keras.Input(shape=(input_size, input_size, 3))
        x = tf.keras.layers.Flatten()(inputs)
        for width in dims[:-1]:
            x = tf.keras.layers.Dense(width, activation="relu")(x)
        x = tf.keras.layers.Dense(dims[-1], activation="sigmoid")(x)
        outputs = tf.keras.layers.Reshape((input_size, input_size, 3))(x)
        return tf.keras.Model(inputs, outputs)

    def _build_conv_autoencoder(self, input_size: int, depth: int):
        tf = self.tf
        channels = {
            2: [32, 3],
            4: [32, 64, 32, 3],
            6: [32, 64, 128, 64, 32, 3],
        }.get(depth)
        if channels is None:
            raise ValueError(f"Unsupported conv autoencoder depth: {depth}")
        inputs = tf.keras.Input(shape=(input_size, input_size, 3))
        x = inputs
        for width in channels[:-1]:
            x = tf.keras.layers.Conv2D(
                width, 3, padding="same", activation="relu"
            )(x)
        outputs = tf.keras.layers.Conv2D(
            channels[-1], 3, padding="same", activation="sigmoid"
        )(x)
        return tf.keras.Model(inputs, outputs)

    def _build_fc_vae(self, input_size: int, beta: float):
        tf = self.tf
        flat = 3 * input_size * input_size
        latent = 64

        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.flatten = tf.keras.layers.Flatten()
                self.enc = tf.keras.layers.Dense(256, activation="relu")
                self.mu = tf.keras.layers.Dense(latent)
                self.logvar = tf.keras.layers.Dense(latent)
                self.dec1 = tf.keras.layers.Dense(256, activation="relu")
                self.dec2 = tf.keras.layers.Dense(flat, activation="sigmoid")
                self.vae_beta = beta
                self.last_kl_loss = None

            def call(self, x, training=False):
                h = self.enc(self.flatten(x))
                mu = self.mu(h)
                logvar = self.logvar(h)
                if training:
                    z = mu + tf.random.normal(tf.shape(mu)) * tf.exp(0.5 * logvar)
                else:
                    z = mu
                self.last_kl_loss = -0.5 * tf.reduce_mean(
                    1.0 + logvar - tf.square(mu) - tf.exp(logvar)
                )
                y = self.dec2(self.dec1(z))
                return tf.reshape(y, (-1, input_size, input_size, 3))

        model = Model()
        model(tf.zeros((1, input_size, input_size, 3)), training=False)
        return model

    def _build_conv_vae(self, input_size: int, beta: float):
        tf = self.tf
        if input_size % 4:
            raise ValueError(
                "Convolutional VAE requires ai.input_size divisible by 4"
            )
        spatial = input_size // 4
        encoded = 64 * spatial * spatial
        latent = 64

        class Model(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.c1 = tf.keras.layers.Conv2D(
                    32, 4, strides=2, padding="same", activation="relu"
                )
                self.c2 = tf.keras.layers.Conv2D(
                    64, 4, strides=2, padding="same", activation="relu"
                )
                self.flatten = tf.keras.layers.Flatten()
                self.mu = tf.keras.layers.Dense(latent)
                self.logvar = tf.keras.layers.Dense(latent)
                self.fc = tf.keras.layers.Dense(encoded, activation="relu")
                self.d1 = tf.keras.layers.Conv2DTranspose(
                    32, 4, strides=2, padding="same", activation="relu"
                )
                self.d2 = tf.keras.layers.Conv2DTranspose(
                    3, 4, strides=2, padding="same", activation="sigmoid"
                )
                self.vae_beta = beta
                self.last_kl_loss = None

            def call(self, x, training=False):
                h = self.flatten(self.c2(self.c1(x)))
                mu = self.mu(h)
                logvar = self.logvar(h)
                if training:
                    z = mu + tf.random.normal(tf.shape(mu)) * tf.exp(0.5 * logvar)
                else:
                    z = mu
                self.last_kl_loss = -0.5 * tf.reduce_mean(
                    1.0 + logvar - tf.square(mu) - tf.exp(logvar)
                )
                y = tf.reshape(self.fc(z), (-1, spatial, spatial, 64))
                return self.d2(self.d1(y))

        model = Model()
        model(tf.zeros((1, input_size, input_size, 3)), training=False)
        return model

    def _build_mlp(self, depth: int):
        tf = self.tf
        ai = self.config["ai"]
        if self.application == "image_classification":
            shape = (int(ai["input_size"]), int(ai["input_size"]), 3)
        elif self.application == "activity_recognition":
            shape = (int(ai["sequence_length"]), int(ai["input_dim"]))
        else:
            raise ValueError(
                "Native MLP supports image_classification or activity_recognition"
            )
        inputs = tf.keras.Input(shape=shape)
        x = tf.keras.layers.Flatten()(inputs)
        for _ in range(depth - 1):
            x = tf.keras.layers.Dense(256, activation="relu")(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        return tf.keras.Model(inputs, outputs)

    def infer(self, array: np.ndarray) -> np.ndarray:
        tf = self.tf
        if self.application in {
            "image_classification",
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }:
            array = np.transpose(array, (0, 2, 3, 1))
            tensor = tf.convert_to_tensor(array, dtype=tf.float32)
        elif self.application == "text_classification":
            tensor = tf.convert_to_tensor(array, dtype=tf.int32)
        else:
            tensor = tf.convert_to_tensor(array, dtype=tf.float32)

        output = self.model(tensor, training=False)
        result = np.asarray(output)
        if self.application in {
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }:
            result = np.transpose(result, (0, 3, 1, 2))
        return result

    def _input_tensor(self, inputs: np.ndarray):
        tf = self.tf
        if self.application in {
            "image_classification",
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }:
            array = np.transpose(inputs, (0, 2, 3, 1))
            return tf.convert_to_tensor(array, dtype=tf.float32)
        if self.application == "text_classification":
            return tf.convert_to_tensor(inputs, dtype=tf.int32)
        return tf.convert_to_tensor(inputs, dtype=tf.float32)

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, Any]:
        tf = self.tf
        x = self._input_tensor(inputs)

        with tf.GradientTape() as tape:
            output = self.model(x, training=True)
            if self.application in {
                "reconstruction",
                "anomaly_detection",
                "image_denoising",
            }:
                target_array = np.transpose(targets, (0, 2, 3, 1))
                y = tf.convert_to_tensor(target_array, dtype=tf.float32)
                reconstruction_loss = tf.reduce_mean(tf.square(output - y))
                mae = tf.reduce_mean(tf.abs(output - y))
                kl_loss = getattr(self.model, "last_kl_loss", None)
                beta = float(getattr(self.model, "vae_beta", 1.0))
                loss = (
                    reconstruction_loss + beta * kl_loss
                    if kl_loss is not None
                    else reconstruction_loss
                )
                accuracy = None
                predictions = None
            else:
                y = tf.reshape(
                    tf.convert_to_tensor(targets, dtype=tf.int64),
                    (-1,),
                )
                losses = tf.keras.losses.sparse_categorical_crossentropy(
                    y,
                    output,
                    from_logits=True,
                )
                loss = tf.reduce_mean(losses)
                predictions = tf.argmax(output, axis=1, output_type=tf.int64)
                accuracy = float(
                    tf.reduce_mean(
                        tf.cast(tf.equal(predictions, y), tf.float32)
                    ).numpy()
                )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.model.trainable_variables
            )
            if gradient is not None
        ]
        self.optimizer.apply_gradients(pairs)

        metrics: Dict[str, Any] = {
            "loss": float(loss.numpy()),
            "learning_rate": self.learning_rate,
            "samples": int(inputs.shape[0]),
        }
        if accuracy is not None:
            metrics["accuracy"] = accuracy
            metrics["_targets"] = np.asarray(y.numpy(), dtype=np.int64)
            metrics["_predictions"] = np.asarray(
                predictions.numpy(), dtype=np.int64
            )
        else:
            metrics["reconstruction_loss"] = float(reconstruction_loss.numpy())
            metrics["mse"] = metrics["reconstruction_loss"]
            metrics["mae"] = float(mae.numpy())
            if kl_loss is not None:
                metrics["kl_loss"] = float(kl_loss.numpy())
                metrics["vae_beta"] = beta
        return metrics

    def evaluation_extras(self) -> Dict[str, Any]:
        kl_loss = getattr(self.model, "last_kl_loss", None)
        if kl_loss is None:
            return {}
        return {
            "kl_loss": float(kl_loss.numpy()),
            "vae_beta": float(getattr(self.model, "vae_beta", 1.0)),
        }

    def get_parameters(self) -> list[np.ndarray]:
        return [np.array(value, copy=True) for value in self.model.get_weights()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        current = self.model.get_weights()
        if len(parameters) != len(current):
            raise ValueError(
                f"Expected {len(current)} parameter tensors, "
                f"received {len(parameters)}"
            )
        for index, (expected, value) in enumerate(zip(current, parameters)):
            if tuple(expected.shape) != tuple(value.shape):
                raise ValueError(
                    f"Parameter {index} shape mismatch: "
                    f"{tuple(value.shape)} != {tuple(expected.shape)}"
                )
        self.model.set_weights(parameters)
        self.optimizer = self.tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )


def _rnn_variant_settings(
    architecture: str,
    variant: str,
) -> tuple[int, bool]:
    prefix = "lstm" if architecture == "lstm" else "gru"
    if variant == f"{prefix}_1layer":
        return 1, False
    if variant == f"{prefix}_2layer":
        return 2, False
    if variant == f"bi{prefix}_2layer":
        return 2, True
    raise ValueError(
        f"Variant {variant!r} is not valid for architecture {architecture!r}"
    )
