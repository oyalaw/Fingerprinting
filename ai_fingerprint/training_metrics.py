from __future__ import annotations

import csv
import datetime as dt
import math
import os
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

    macro_precision = float(mean(precision_values))
    macro_recall = float(mean(recall_values))
    macro_f1 = float(mean(f1_values))
    return {
        "accuracy": float(np.mean(truth == pred)),
        # Backward-compatible aliases. New analysis should use the explicit
        # macro_* names for multiclass classification.
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "evaluated_samples": int(truth.size),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def binary_anomaly_metrics(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Binary anomaly metrics with anomaly=1 as the positive class."""
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.size != score.size:
        raise ValueError("Anomaly truth and score arrays must have equal length")
    if truth.size == 0:
        return {}
    pred = (score > float(threshold)).astype(np.int64)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / truth.size

    positives = int(np.sum(truth == 1))
    negatives = int(np.sum(truth == 0))
    auroc = None
    if positives and negatives:
        ranks = _average_ranks(score)
        rank_sum_positive = float(np.sum(ranks[truth == 1]))
        auroc = (
            rank_sum_positive - positives * (positives + 1) / 2.0
        ) / (positives * negatives)

    auprc = None
    if positives:
        order = np.argsort(-score, kind="mergesort")
        sorted_truth = truth[order]
        cumulative_tp = np.cumsum(sorted_truth == 1)
        positions = np.arange(1, truth.size + 1, dtype=np.float64)
        precision_at_k = cumulative_tp / positions
        auprc = float(np.sum(precision_at_k[sorted_truth == 1]) / positives)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc) if auroc is not None else None,
        "auprc": float(auprc) if auprc is not None else None,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "evaluated_samples": int(truth.size),
        "anomaly_samples": positives,
        "normal_samples": negatives,
    }


def collect_anomaly_batches(
    generator,
    batches: int,
    *,
    normal_only: bool | None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(max(int(batches), 1)):
        inputs, targets, labels = generator.anomaly_batch(normal_only=normal_only)
        result.append((
            np.asarray(inputs),
            np.asarray(targets),
            np.asarray(labels, dtype=np.int64),
        ))
    return result


def _reconstruction_scores(workload, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predicted = np.asarray(workload.infer(inputs), dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(
            f"Anomaly evaluation output shape {predicted.shape} does not match target {target.shape}"
        )
    error = np.square(predicted - target)
    axes = tuple(range(1, error.ndim))
    if not axes:
        return error.reshape(-1)
    return np.mean(error, axis=axes).reshape(-1)


def evaluate_anomaly_batches(
    workload,
    calibration_batches: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    evaluation_batches: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    threshold_percentile: float = 95.0,
) -> tuple[Dict[str, Any], float]:
    """Evaluate reconstruction-based anomaly detection on fixed held-out data.

    The threshold is recomputed for the current model from held-out normal
    calibration samples, then applied to a separate normal+anomaly evaluation
    set. This allows accuracy/precision/recall/F1/AUROC/AUPRC to be logged per
    FL round without using anomaly labels as training targets.
    """
    started = time.perf_counter_ns()
    calibration_scores: list[np.ndarray] = []
    for inputs, targets, labels in calibration_batches:
        scores = _reconstruction_scores(workload, inputs, targets)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        normal = scores[labels == 0]
        if normal.size:
            calibration_scores.append(normal)
    if not calibration_scores:
        raise ValueError("Anomaly calibration set contains no normal examples")
    calibration = np.concatenate(calibration_scores)
    percentile = float(threshold_percentile)
    threshold = float(np.percentile(calibration, percentile))

    rows: list[Dict[str, Any]] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for inputs, targets, labels in evaluation_batches:
        rows.append(dict(workload.evaluate_batch(inputs, targets)))
        all_scores.append(_reconstruction_scores(workload, inputs, targets))
        all_labels.append(np.asarray(labels, dtype=np.int64).reshape(-1))
    reconstruction = aggregate_batch_metrics(rows)
    scores = np.concatenate(all_scores) if all_scores else np.asarray([], dtype=np.float64)
    labels = np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.int64)
    binary = binary_anomaly_metrics(labels, scores, threshold)
    result = dict(reconstruction)
    result.update({
        "accuracy": binary.get("accuracy"),
        "precision": binary.get("precision"),
        "recall": binary.get("recall"),
        "f1": binary.get("f1"),
        "anomaly_accuracy": binary.get("accuracy"),
        "anomaly_precision": binary.get("precision"),
        "anomaly_recall": binary.get("recall"),
        "anomaly_f1": binary.get("f1"),
        "anomaly_auroc": binary.get("auroc"),
        "anomaly_auprc": binary.get("auprc"),
        "anomaly_tp": binary.get("tp"),
        "anomaly_fp": binary.get("fp"),
        "anomaly_tn": binary.get("tn"),
        "anomaly_fn": binary.get("fn"),
        "anomaly_threshold": threshold,
        "anomaly_threshold_percentile": percentile,
        "anomaly_eval_samples": binary.get("evaluated_samples"),
        "anomaly_samples": binary.get("anomaly_samples"),
        "normal_samples": binary.get("normal_samples"),
        "normal_error_mean": float(np.mean(scores[labels == 0])) if np.any(labels == 0) else None,
        "anomaly_error_mean": float(np.mean(scores[labels == 1])) if np.any(labels == 1) else None,
        "evaluated_samples": binary.get("evaluated_samples"),
    })
    ended = time.perf_counter_ns()
    return result, (ended - started) / 1_000_000.0


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
    "partition_type", "partition_alpha", "partition_seed", "partition_index",
    "partition_disjoint", "partition_sample_count",
    "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1",
    "train_macro_precision", "train_macro_recall", "train_macro_f1",
    "train_reconstruction_loss", "train_mse", "train_mae", "train_kl_loss", "vae_beta",
    "anomaly_accuracy", "anomaly_precision", "anomaly_recall", "anomaly_f1",
    "anomaly_auroc", "anomaly_auprc", "anomaly_threshold",
    "anomaly_threshold_percentile", "anomaly_tp", "anomaly_fp", "anomaly_tn",
    "anomaly_fn", "anomaly_eval_samples", "anomaly_samples", "normal_samples",
    "normal_error_mean", "anomaly_error_mean",
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
    "training_policy_id", "policy_source", "partition_type", "partition_alpha",
    "partition_seed", "partition_disjoint", "clients_expected",
    "clients_received", "client_ids", "total_examples", "aggregation_time_ms",
    "evaluation_time_ms", "global_loss", "global_accuracy", "global_precision",
    "global_recall", "global_f1", "global_macro_precision", "global_macro_recall",
    "global_macro_f1", "global_reconstruction_loss", "global_mse",
    "global_mae", "global_kl_loss", "vae_beta", "evaluation_samples",
    "anomaly_accuracy", "anomaly_precision", "anomaly_recall", "anomaly_f1",
    "anomaly_auroc", "anomaly_auprc", "anomaly_threshold",
    "anomaly_threshold_percentile", "anomaly_tp", "anomaly_fp", "anomaly_tn",
    "anomaly_fn", "anomaly_eval_samples", "anomaly_samples", "normal_samples",
    "normal_error_mean", "anomaly_error_mean",
    "evaluation_split", "global_model_norm_l2", "global_update_norm_l2",
    "mean_client_update_norm_l2", "std_client_update_norm_l2", "model_size_bytes",
    "bytes_received_round", "bytes_sent_round", "round_duration_sec",
    "loss_change", "accuracy_change", "f1_change", "timestamp_start_utc",
    "timestamp_end_utc", "evaluation_error",
]

SERVER_CLIENT_UPDATE_FIELDS = [
    "experiment_id", "round", "client_id", "num_examples", "payload_bytes",
    "partition_type", "partition_alpha", "partition_seed", "partition_index",
    "partition_disjoint", "partition_sample_count", "partition_assignment_id",
    "client_loss", "client_accuracy", "client_precision", "client_recall", "client_f1",
    "client_macro_precision", "client_macro_recall", "client_macro_f1",
    "client_reconstruction_loss", "client_mse", "client_mae", "client_kl_loss",
    "vae_beta", "client_anomaly_accuracy", "client_anomaly_precision",
    "client_anomaly_recall", "client_anomaly_f1", "client_anomaly_auroc",
    "client_anomaly_auprc", "client_anomaly_threshold", "client_anomaly_tp",
    "client_anomaly_fp", "client_anomaly_tn", "client_anomaly_fn", "client_model_norm_l2", "client_update_norm_l2",
    "receive_timestamp_utc",
]

ANOMALY_METRIC_FIELDS = [
    "experiment_id", "role", "client_id", "round", "family", "architecture",
    "variant", "application", "dataset", "framework", "partition_type",
    "partition_alpha", "partition_seed", "loss", "accuracy", "precision",
    "recall", "f1_score", "auroc", "auprc", "threshold",
    "threshold_percentile", "tp", "fp", "tn", "fn", "evaluation_samples",
    "anomaly_samples", "normal_samples", "reconstruction_loss", "mse", "mae",
    "normal_error_mean", "anomaly_error_mean", "timestamp_start_utc",
    "timestamp_end_utc",
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
                handle.flush()
                os.fsync(handle.fileno())
        return path

    def _partition_token(self) -> str:
        partition = self.config.get("data", {}).get("partition", {}) or {}
        kind = str(partition.get("type", "iid")).strip().lower()
        if kind == "non_iid":
            alpha = str(partition.get("alpha", 0.5)).replace(".", "p")
            return f"non_iid_alpha_{alpha}"
        return "iid"

    def _write_anomaly_summary(self, row: Mapping[str, Any], *, role: str) -> None:
        if role == "client":
            prefix = "train_"
            loss = row.get("train_loss")
            reconstruction_loss = row.get("train_reconstruction_loss")
            mse = row.get("train_mse")
            mae = row.get("train_mae")
        else:
            prefix = "global_"
            loss = row.get("global_loss")
            reconstruction_loss = row.get("global_reconstruction_loss")
            mse = row.get("global_mse")
            mae = row.get("global_mae")
        summary = {
            "experiment_id": row.get("experiment_id"),
            "role": role,
            "client_id": row.get("client_id") if role == "client" else None,
            "round": row.get("round"),
            "family": row.get("family"),
            "architecture": row.get("architecture"),
            "variant": row.get("variant"),
            "application": row.get("application"),
            "dataset": row.get("dataset"),
            "framework": row.get("framework"),
            "partition_type": row.get("partition_type"),
            "partition_alpha": row.get("partition_alpha"),
            "partition_seed": row.get("partition_seed"),
            "loss": loss,
            "accuracy": row.get("anomaly_accuracy", row.get(prefix + "accuracy")),
            "precision": row.get("anomaly_precision", row.get(prefix + "precision")),
            "recall": row.get("anomaly_recall", row.get(prefix + "recall")),
            "f1_score": row.get("anomaly_f1", row.get(prefix + "f1")),
            "auroc": row.get("anomaly_auroc"),
            "auprc": row.get("anomaly_auprc"),
            "threshold": row.get("anomaly_threshold"),
            "threshold_percentile": row.get("anomaly_threshold_percentile"),
            "tp": row.get("anomaly_tp"),
            "fp": row.get("anomaly_fp"),
            "tn": row.get("anomaly_tn"),
            "fn": row.get("anomaly_fn"),
            "evaluation_samples": row.get("anomaly_eval_samples"),
            "anomaly_samples": row.get("anomaly_samples"),
            "normal_samples": row.get("normal_samples"),
            "reconstruction_loss": reconstruction_loss,
            "mse": mse,
            "mae": mae,
            "normal_error_mean": row.get("normal_error_mean"),
            "anomaly_error_mean": row.get("anomaly_error_mean"),
            "timestamp_start_utc": row.get("timestamp_start_utc"),
            "timestamp_end_utc": row.get("timestamp_end_utc"),
        }
        self._write("anomaly_detection_metrics.csv", ANOMALY_METRIC_FIELDS, summary)
        self._write(
            f"anomaly_detection_metrics_{self._partition_token()}.csv",
            ANOMALY_METRIC_FIELDS,
            summary,
        )

    def write_client_round(self, row: Mapping[str, Any]) -> Path:
        path = self._write("round_metrics.csv", CLIENT_ROUND_FIELDS, row)
        self._write(
            f"round_metrics_{self._partition_token()}.csv",
            CLIENT_ROUND_FIELDS,
            row,
        )
        if str(row.get("application", "")) == "anomaly_detection":
            self._write("anomaly_metrics.csv", CLIENT_ROUND_FIELDS, row)
            self._write(
                f"anomaly_metrics_{self._partition_token()}.csv",
                CLIENT_ROUND_FIELDS,
                row,
            )
            self._write_anomaly_summary(row, role="client")
        return path

    def write_server_round(self, row: Mapping[str, Any]) -> Path:
        path = self._write("round_metrics.csv", SERVER_ROUND_FIELDS, row)
        self._write(
            f"round_metrics_{self._partition_token()}.csv",
            SERVER_ROUND_FIELDS,
            row,
        )
        if str(row.get("application", "")) == "anomaly_detection":
            self._write("anomaly_metrics.csv", SERVER_ROUND_FIELDS, row)
            self._write(
                f"anomaly_metrics_{self._partition_token()}.csv",
                SERVER_ROUND_FIELDS,
                row,
            )
            self._write_anomaly_summary(row, role="server")
        return path

    def write_server_client_update(self, row: Mapping[str, Any]) -> Path:
        path = self._write(
            "client_update_metrics.csv",
            SERVER_CLIENT_UPDATE_FIELDS,
            row,
        )
        self._write(
            f"client_update_metrics_{self._partition_token()}.csv",
            SERVER_CLIENT_UPDATE_FIELDS,
            row,
        )
        return path


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
        "macro_precision", "macro_recall", "macro_f1",
        "reconstruction_loss", "mse", "mae", "kl_loss", "vae_beta",
        "anomaly_accuracy", "anomaly_precision", "anomaly_recall", "anomaly_f1",
        "anomaly_auroc", "anomaly_auprc", "anomaly_threshold",
        "anomaly_tp", "anomaly_fp", "anomaly_tn", "anomaly_fn",
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
