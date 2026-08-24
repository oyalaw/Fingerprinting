from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Workload


class ONNXRuntimeWorkload(Workload):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        artifact = config["ai"]["model_artifact"]
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime requested but onnxruntime is not installed"
            ) from exc

        providers = [
            provider
            for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider in ort.get_available_providers()
        ]
        self.session = ort.InferenceSession(artifact, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, array: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: array})
        return np.asarray(outputs[0])
