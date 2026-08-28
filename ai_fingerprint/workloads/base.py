from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import math

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
    ) -> Dict[str, Any]:
        raise RuntimeError(
            f"{type(self).__name__} does not support training"
        )


    def evaluation_extras(self) -> Dict[str, Any]:
        """Optional backend-specific evaluation metrics (for example VAE KL)."""
        return {}

    def evaluate_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate one batch without updating model parameters.

        Classification returns cross-entropy loss plus predictions/targets for
        exact round-level macro metrics. Reconstruction workloads return MSE
        and MAE. Backend-specific extras can add KL loss and beta for VAEs.
        """
        output = np.asarray(self.infer(inputs))
        application = str(self.config["ai"]["application"])
        samples = int(np.asarray(inputs).shape[0])

        if application in {
            "reconstruction",
            "anomaly_detection",
            "image_denoising",
        }:
            target = np.asarray(targets, dtype=np.float64)
            predicted = output.astype(np.float64, copy=False)
            if predicted.shape != target.shape:
                raise ValueError(
                    f"Evaluation output shape {predicted.shape} does not match "
                    f"target shape {target.shape}"
                )
            error = predicted - target
            mse = float(np.mean(np.square(error)))
            mae = float(np.mean(np.abs(error)))
            metrics: Dict[str, Any] = {
                "loss": mse,
                "reconstruction_loss": mse,
                "mse": mse,
                "mae": mae,
                "samples": samples,
            }
            extras = dict(self.evaluation_extras() or {})
            kl = extras.get("kl_loss")
            beta = extras.get("vae_beta")
            if kl is not None:
                beta_value = float(beta if beta is not None else 1.0)
                metrics["kl_loss"] = float(kl)
                metrics["vae_beta"] = beta_value
                metrics["loss"] = mse + beta_value * float(kl)
            return metrics

        logits = output.astype(np.float64, copy=False)
        if logits.ndim != 2:
            raise ValueError(
                f"Classification evaluation expects [batch, classes] logits, "
                f"received shape {logits.shape}"
            )
        truth = np.asarray(targets, dtype=np.int64).reshape(-1)
        if truth.size != logits.shape[0]:
            raise ValueError(
                f"Classification target count {truth.size} does not match "
                f"batch size {logits.shape[0]}"
            )
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
        correct = shifted[np.arange(truth.size), truth]
        loss = float(np.mean(log_sum_exp - correct))
        predictions = np.argmax(logits, axis=1).astype(np.int64)
        return {
            "loss": loss,
            "accuracy": float(np.mean(predictions == truth)),
            "_targets": truth,
            "_predictions": predictions,
            "samples": samples,
        }

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
