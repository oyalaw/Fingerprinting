from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class Workload(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def infer(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, float]:
        raise RuntimeError(
            f"{type(self).__name__} does not support training"
        )

    def get_parameters(self) -> list[np.ndarray]:
        raise RuntimeError(
            f"{type(self).__name__} does not expose trainable parameters"
        )

    def set_parameters(
        self,
        parameters: list[np.ndarray],
    ) -> None:
        raise RuntimeError(
            f"{type(self).__name__} does not expose trainable parameters"
        )


def build_workload(config: Dict[str, Any]) -> Workload:
    runtime = config["ai"]["runtime"]
    framework = config["ai"]["framework"]

    if runtime == "native" and framework == "pytorch":
        from .pytorch_backend import PyTorchWorkload
        return PyTorchWorkload(config)

    if runtime == "native" and framework == "tensorflow":
        from .tensorflow_backend import TensorFlowWorkload
        return TensorFlowWorkload(config)

    if runtime == "tflite":
        from .tflite_backend import TFLiteWorkload
        return TFLiteWorkload(config)

    if runtime == "onnxruntime":
        from .onnx_backend import ONNXRuntimeWorkload
        return ONNXRuntimeWorkload(config)

    if runtime == "tensorrt":
        from .tensorrt_backend import TensorRTWorkload
        return TensorRTWorkload(config)

    raise ValueError(
        f"No workload backend for framework={framework!r}, runtime={runtime!r}"
    )
