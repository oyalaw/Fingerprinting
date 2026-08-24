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
            if variant != "tiny_transformer_2layer":
                raise ValueError(
                    f"Unsupported native tiny transformer variant: "
                    f"{variant!r}"
                )
            return self._build_tiny_transformer()

        if arch == "convolutional_autoencoder":
            if variant != "convolutional_autoencoder_4layer":
                raise ValueError(
                    f"Unsupported native autoencoder variant: "
                    f"{variant!r}"
                )
            return self._build_conv_autoencoder(input_size)

        raise ValueError(
            f"Native TensorFlow builder not implemented for "
            f"architecture={arch!r}, variant={variant!r}. "
            "Use a supported native variant or an artifact backed runtime."
        )

    def _rnn_layer(
        self,
        layer_cls,
        return_sequences: bool,
        bidirectional: bool,
    ):
        layer = layer_cls(
            128,
            return_sequences=return_sequences,
        )
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
        layer_cls = (
            tf.keras.layers.LSTM
            if kind == "lstm"
            else tf.keras.layers.GRU
        )

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
        layer_cls = (
            tf.keras.layers.LSTM
            if kind == "lstm"
            else tf.keras.layers.GRU
        )

        for index in range(num_layers):
            x = self._rnn_layer(
                layer_cls=layer_cls,
                return_sequences=index < num_layers - 1,
                bidirectional=bidirectional,
            )(x)

        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        return tf.keras.Model(inputs, outputs)

    def _build_tiny_transformer(self):
        tf = self.tf
        length = int(self.config["ai"]["max_text_length"])
        vocab = int(self.config["ai"]["vocab_size"])
        embed = 128

        inputs = tf.keras.Input(shape=(length,), dtype="int32")
        x = tf.keras.layers.Embedding(vocab, embed)(inputs)

        for _ in range(2):
            attn = tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=32,
            )(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attn)
            ff = tf.keras.layers.Dense(
                256,
                activation="relu",
            )(x)
            ff = tf.keras.layers.Dense(embed)(ff)
            x = tf.keras.layers.LayerNormalization()(x + ff)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        return tf.keras.Model(inputs, outputs)

    def _build_conv_autoencoder(self, input_size: int):
        tf = self.tf
        inputs = tf.keras.Input(
            shape=(input_size, input_size, 3)
        )
        x = tf.keras.layers.Conv2D(
            32, 3, padding="same", activation="relu"
        )(inputs)
        x = tf.keras.layers.Conv2D(
            64, 3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            32, 3, padding="same", activation="relu"
        )(x)
        outputs = tf.keras.layers.Conv2D(
            3, 3, padding="same", activation="sigmoid"
        )(x)
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
            tensor = tf.convert_to_tensor(
                array,
                dtype=tf.float32,
            )
        elif self.application == "text_classification":
            tensor = tf.convert_to_tensor(
                array,
                dtype=tf.int32,
            )
        else:
            tensor = tf.convert_to_tensor(
                array,
                dtype=tf.float32,
            )

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
            inputs = np.transpose(inputs, (0, 2, 3, 1))
            return tf.convert_to_tensor(inputs, dtype=tf.float32)
        if self.application == "text_classification":
            return tf.convert_to_tensor(inputs, dtype=tf.int32)
        return tf.convert_to_tensor(inputs, dtype=tf.float32)

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        tf = self.tf
        x = self._input_tensor(inputs)

        reconstruction = self.application in {
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }
        if reconstruction:
            targets = np.transpose(targets, (0, 2, 3, 1))
            y = tf.convert_to_tensor(targets, dtype=tf.float32)
        else:
            y = tf.convert_to_tensor(
                np.asarray(targets).reshape(-1),
                dtype=tf.int64,
            )

        with tf.GradientTape() as tape:
            output = self.model(x, training=True)
            if reconstruction:
                loss = tf.reduce_mean(
                    tf.math.squared_difference(output, y)
                )
            else:
                losses = tf.keras.losses.sparse_categorical_crossentropy(
                    y,
                    output,
                    from_logits=True,
                )
                loss = tf.reduce_mean(losses)

        gradients = tape.gradient(
            loss,
            self.model.trainable_variables,
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        metrics: Dict[str, float] = {
            "loss": float(loss.numpy()),
            "learning_rate": float(
                tf.keras.backend.get_value(
                    self.optimizer.learning_rate
                )
            ),
        }

        if not reconstruction:
            predictions = tf.argmax(
                output,
                axis=1,
                output_type=tf.int64,
            )
            accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(predictions, y),
                    tf.float32,
                )
            )
            metrics["accuracy"] = float(accuracy.numpy())

        return metrics

    def get_parameters(self) -> list[np.ndarray]:
        return [
            np.asarray(value).copy()
            for value in self.model.get_weights()
        ]

    def set_parameters(
        self,
        parameters: list[np.ndarray],
    ) -> None:
        current = self.model.get_weights()
        if len(parameters) != len(current):
            raise ValueError(
                f"Expected {len(current)} parameter tensors, "
                f"received {len(parameters)}"
            )
        for index, (received, expected) in enumerate(
            zip(parameters, current)
        ):
            if tuple(received.shape) != tuple(expected.shape):
                raise ValueError(
                    f"Parameter {index} shape mismatch: "
                    f"{tuple(received.shape)} != {tuple(expected.shape)}"
                )
        self.model.set_weights(parameters)
        self.optimizer = self.tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )


def _rnn_variant_settings(
    architecture: str,
    variant: str,
) -> tuple[int, bool]:
    expected_prefix = "lstm" if architecture == "lstm" else "gru"

    if variant == f"{expected_prefix}_1layer":
        return 1, False
    if variant == f"{expected_prefix}_2layer":
        return 2, False
    if variant == f"bi{expected_prefix}_2layer":
        return 2, True

    raise ValueError(
        f"Variant {variant!r} is not valid for architecture "
        f"{architecture!r}"
    )
