from __future__ import annotations

import csv
import datetime as dt
import math
import threading
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parameter_l2_norm(parameters: Sequence[np.ndarray]) -> float:
    """Return a numerically stable L2 norm across a model parameter list."""
    squared = 0.0
    for value in parameters:
        array = np.asarray(value)
        if array.size == 0:
            continue
        flat = array.astype(np.float64, copy=False).ravel()
        squared += float(np.dot(flat, flat))
    return float(math.sqrt(max(squared, 0.0)))


def parameter_delta_l2_norm(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
) -> float:
    """Return ||left-right||_2 across corresponding parameter tensors."""
    if len(left) != len(right):
        raise ValueError(
            f"Parameter tensor-count mismatch: {len(left)} != {len(right)}"
        )
    squared = 0.0
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        a = np.asarray(lhs)
        b = np.asarray(rhs)
        if a.shape != b.shape:
            raise ValueError(
                f"Parameter tensor {index} shape mismatch: {a.shape} != {b.shape}"
            )
        if a.size == 0:
            continue
        delta = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
        flat = delta.ravel()
        squared += float(np.dot(flat, flat))
    return float(math.sqrt(max(squared, 0.0)))


def _safe_mean(values: Iterable[Any]) -> float | None:
    parsed: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parsed.append(number)
    return float(mean(parsed)) if parsed else None


def _safe_pstdev(values: Iterable[Any]) -> float | None:
    parsed: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parsed.append(number)
    if not parsed:
        return None
    return float(pstdev(parsed)) if len(parsed) > 1 else 0.0


def classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> Dict[str, float]:
    """
    Compute accuracy and macro precision/recall/F1 without sklearn.

    Macro averaging follows the labels observed in y_true or y_pred. This is
    preferable for small per-round samples because absent classes are not
    artificially assigned zero scores merely because they were not sampled.
    """
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.size != pred.size:
        raise ValueError("y_true and y_pred must contain the same number of samples")
    if truth.size == 0:
        return {}

    labels = sorted(set(truth.tolist()) | set(pred.tolist()))
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        tp = int(np.sum((truth == label) & (pred == label)))
        fp = int(np.sum((truth != label) & (pred == label)))
        fn = int(np.sum((truth == label) & (pred != label)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))

    return {
        "accuracy": float(np.mean(truth == pred)),
        "precision": float(mean(precision_values)),
        "recall": float(mean(recall_values)),
        "f1": float(mean(f1_values)),
        "evaluated_samples": int(truth.size),
    }


def aggregate_batch_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate workload batch metrics into one round/evaluation result."""
    if not rows:
        return {}

    result: Dict[str, Any] = {}
    for field in (
        "loss",
        "reconstruction_loss",
        "mse",
        "mae",
        "kl_loss",
        "vae_beta",
        "learning_rate",
    ):
        value = _safe_mean(row.get(field) for row in rows)
        if value is not None:
            result[field] = value

    truth_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for row in rows:
        truth = row.get("_targets")
        pred = row.get("_predictions")
        if truth is None or pred is None:
            continue
        truth_parts.append(np.asarray(truth, dtype=np.int64).reshape(-1))
        pred_parts.append(np.asarray(pred, dtype=np.int64).reshape(-1))

    if truth_parts:
        class_metrics = classification_metrics(
            np.concatenate(truth_parts),
            np.concatenate(pred_parts),
        )
        result.update(class_metrics)
    else:
        value = _safe_mean(row.get("accuracy") for row in rows)
        if value is not None:
            result["accuracy"] = value

    sample_total = 0
    for row in rows:
        value = row.get("samples")
        if value is not None:
            sample_total += int(value)
    if sample_total:
        result["evaluated_samples"] = sample_total

    return result


def evaluate_generator(
    workload,
    generator,
    batches: int,
) -> tuple[Dict[str, Any], float]:
    """Evaluate a workload for a fixed number of batches and return metrics/time."""
    rows: list[Dict[str, Any]] = []
    started = time.perf_counter_ns()
    for _ in range(max(int(batches), 1)):
        inputs, targets = generator.training_batch()
        row = dict(workload.evaluate_batch(inputs, targets))
        row.setdefault("samples", int(inputs.shape[0]))
        rows.append(row)
    ended = time.perf_counter_ns()
    return aggregate_batch_metrics(rows), (ended - started) / 1_000_000.0


CLIENT_ROUND_FIELDS = [
    "experiment_id", "client_id", "round", "family", "architecture", "variant",
    "application", "dataset", "framework", "runtime", "precision", "input_size",
    "batch_size", "global_rounds", "local_epochs", "steps_per_epoch", "local_steps",
    "train_samples", "learning_rate", "training_policy_id", "policy_source",
    "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1",
    "train_reconstruction_loss", "train_mse", "train_mae", "train_kl_loss", "vae_beta",
    "train_loss_before", "train_loss_after", "loss_change",
    "accuracy_before", "accuracy_after", "accuracy_change",
    "f1_before", "f1_after", "f1_change",
    "mse_before", "mse_after", "mse_change",
    "round_probe_samples", "metric_source",
    "download_time_sec", "training_time_sec", "upload_time_sec", "sync_wait_time_sec",
    "transaction_time_sec", "model_size_bytes", "global_model_norm_l2",
    "local_model_norm_l2", "update_norm_l2", "timestamp_start_utc", "timestamp_end_utc",
]

SERVER_ROUND_FIELDS = [
    "experiment_id", "round", "family", "architecture", "variant", "application",
    "dataset", "framework", "runtime", "aggregation_rule", "input_size", "batch_size",
    "learning_rate", "global_rounds", "local_epochs", "steps_per_epoch",
    "training_policy_id", "policy_source", "clients_expected",
    "clients_received", "client_ids", "total_examples", "aggregation_time_ms",
    "evaluation_time_ms", "global_loss", "global_accuracy", "global_precision",
    "global_recall", "global_f1", "global_reconstruction_loss", "global_mse",
    "global_mae", "global_kl_loss", "vae_beta", "evaluation_samples",
    "evaluation_split", "global_model_norm_l2", "global_update_norm_l2",
    "mean_client_update_norm_l2", "std_client_update_norm_l2", "model_size_bytes",
    "bytes_received_round", "bytes_sent_round", "round_duration_sec",
    "loss_change", "accuracy_change", "f1_change", "timestamp_start_utc",
    "timestamp_end_utc", "evaluation_error",
]

SERVER_CLIENT_UPDATE_FIELDS = [
    "experiment_id", "round", "client_id", "num_examples", "payload_bytes",
    "client_loss", "client_accuracy", "client_precision", "client_recall", "client_f1",
    "client_reconstruction_loss", "client_mse", "client_mae", "client_kl_loss",
    "vae_beta", "client_model_norm_l2", "client_update_norm_l2",
    "receive_timestamp_utc",
]


class PerformanceLogWriter:
    """Thread-safe CSV writer for per-round training-performance artifacts."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _write(self, name: str, fieldnames: Sequence[str], row: Mapping[str, Any]) -> Path:
        path = self.output_dir / name
        with self._lock:
            is_new = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
                if is_new:
                    writer.writeheader()
                safe_row = {key: row.get(key) for key in fieldnames}
                writer.writerow(safe_row)
        return path

    def write_client_round(self, row: Mapping[str, Any]) -> Path:
        return self._write("round_metrics.csv", CLIENT_ROUND_FIELDS, row)

    def write_server_round(self, row: Mapping[str, Any]) -> Path:
        return self._write("round_metrics.csv", SERVER_ROUND_FIELDS, row)

    def write_server_client_update(self, row: Mapping[str, Any]) -> Path:
        return self._write(
            "client_update_metrics.csv",
            SERVER_CLIENT_UPDATE_FIELDS,
            row,
        )


def metric_delta(current: Any, previous: Any, *, improvement_direction: str = "up") -> float | None:
    if current is None or previous is None:
        return None
    try:
        current_f = float(current)
        previous_f = float(previous)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(current_f) and math.isfinite(previous_f)):
        return None
    if improvement_direction == "down":
        return previous_f - current_f
    return current_f - previous_f


def scalar_metrics_for_wire(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Return only JSON-safe scalar performance metrics for FL headers."""
    allowed = {
        "loss", "accuracy", "precision", "recall", "f1",
        "reconstruction_loss", "mse", "mae", "kl_loss", "vae_beta",
    }
    result: Dict[str, Any] = {}
    for key in allowed:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[key] = number
    return result


def mean_std_update_norms(updates: Sequence[Any]) -> tuple[float | None, float | None]:
    values = [getattr(update, "metrics", {}).get("update_norm_l2") for update in updates]
    return _safe_mean(values), _safe_pstdev(values)
