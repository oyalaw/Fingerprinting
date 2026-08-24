from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Workload


class TFLiteWorkload(Workload):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        artifact = config["ai"]["model_artifact"]
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow Lite runtime requested but tensorflow is not installed"
            ) from exc

        self.tf = tf
        self.interpreter = tf.lite.Interpreter(model_path=artifact)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def infer(self, array: np.ndarray) -> np.ndarray:
        detail = self.input_details[0]
        target_dtype = detail["dtype"]
        value = array.astype(target_dtype, copy=False)

        expected_shape = tuple(int(v) for v in detail["shape"])
        if value.shape != expected_shape:
            try:
                self.interpreter.resize_tensor_input(detail["index"], value.shape)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                detail = self.input_details[0]
            except Exception as exc:
                raise ValueError(
                    f"Input shape {value.shape} does not match TFLite model shape "
                    f"{expected_shape}"
                ) from exc

        self.interpreter.set_tensor(detail["index"], value)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])
