from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ai_fingerprint.config import DEFAULT_CONFIG
from ai_fingerprint.dataset_manager import DatasetManager
from ai_fingerprint.federated_policy import build_training_policy, apply_training_policy
from ai_fingerprint.training_metrics import (
    PerformanceLogWriter,
    binary_anomaly_metrics,
    evaluate_anomaly_batches,
)


class _DummyWorkload:
    def infer(self, array):
        return np.asarray(array, dtype=np.float32)

    def evaluate_batch(self, inputs, targets):
        error = np.asarray(inputs, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
        mse = float(np.mean(np.square(error)))
        mae = float(np.mean(np.abs(error)))
        return {
            "loss": mse,
            "reconstruction_loss": mse,
            "mse": mse,
            "mae": mae,
            "samples": int(np.asarray(inputs).shape[0]),
        }


def test_default_federated_rounds_are_fullscale_100():
    assert DEFAULT_CONFIG["federated"]["rounds"] == 100


def test_binary_anomaly_metrics_populates_accuracy_precision_recall_f1():
    metrics = binary_anomaly_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        threshold=0.5,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0


def test_anomaly_evaluation_uses_normal_calibration_and_populates_metrics():
    # Per-sample reconstruction MSE is the square of the scalar input because
    # the reconstruction target is all zeros.
    calibration = [(
        np.asarray([[[[0.1]]], [[[0.2]]]], dtype=np.float32),
        np.zeros((2, 1, 1, 1), dtype=np.float32),
        np.asarray([0, 0], dtype=np.int64),
    )]
    evaluation = [(
        np.asarray([[[[0.1]]], [[[0.15]]], [[[0.9]]], [[[1.0]]]], dtype=np.float32),
        np.zeros((4, 1, 1, 1), dtype=np.float32),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )]
    metrics, _ = evaluate_anomaly_batches(
        _DummyWorkload(),
        calibration,
        evaluation,
        threshold_percentile=95.0,
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["anomaly_auroc"] == 1.0
    assert metrics["anomaly_auprc"] == 1.0
    assert metrics["anomaly_threshold"] > 0


def _anomaly_synthetic_config(tmp_path: Path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["output_dir"] = str(tmp_path)
    config["node"]["role"] = "client"
    config["execution"].update({
        "task": "training",
        "deployment": "federated",
        "batch_size": 4,
        "seed": 42,
    })
    config["federated"].update({"expected_clients": 3})
    config["ai"].update({
        "family": "autoencoder",
        "architecture": "convolutional_autoencoder",
        "variant": "convolutional_autoencoder_2layer",
        "application": "anomaly_detection",
        "dataset": "synthetic_image",
        "num_classes": 10,
        "input_size": 8,
    })
    config["data"].update({"split": "train", "shuffle": False})
    config["data"]["partition"].update({
        "type": "iid",
        "client_count": 3,
        "client_index": 0,
        "seed": 42,
    })
    config["anomaly_detection"]["anomaly_labels"] = [9]
    return config


def test_anomaly_training_excludes_configured_anomaly_class(tmp_path):
    config = _anomaly_synthetic_config(tmp_path)
    manager = DatasetManager(config)
    labels = [manager._partition_label(int(index)) for index in manager._indices]
    assert labels
    assert 9 not in labels


def test_policy_distributes_anomaly_protocol_to_client(tmp_path):
    server = _anomaly_synthetic_config(tmp_path)
    server["node"]["role"] = "server"
    server["federated"]["rounds"] = 100
    server["anomaly_detection"].update({
        "anomaly_labels": [8, 9],
        "threshold_percentile": 97.5,
        "calibration_batches": 7,
        "evaluation_batches": 11,
        "evaluation_batch_size": 16,
    })
    policy = build_training_policy(server)
    client = _anomaly_synthetic_config(tmp_path / "client")
    client["experiment"]["experiment_id"] = server["experiment"]["experiment_id"]
    apply_training_policy(client, policy)
    assert client["federated"]["rounds"] == 100
    assert client["anomaly_detection"]["anomaly_labels"] == [8, 9]
    assert client["anomaly_detection"]["threshold_percentile"] == 97.5
    assert client["anomaly_detection"]["evaluation_batch_size"] == 16


def test_anomaly_rounds_get_separate_metric_files(tmp_path):
    config = _anomaly_synthetic_config(tmp_path)
    writer = PerformanceLogWriter(config)
    row = {
        "application": "anomaly_detection",
        "partition_type": "iid",
        "train_loss": 0.1,
        "train_accuracy": 0.9,
        "train_precision": 0.8,
        "train_recall": 0.7,
        "train_f1": 0.7467,
    }
    writer.write_client_round(row)
    assert (tmp_path / "round_metrics.csv").exists()
    assert (tmp_path / "round_metrics_iid.csv").exists()
    assert (tmp_path / "anomaly_metrics.csv").exists()
    assert (tmp_path / "anomaly_metrics_iid.csv").exists()
    concise = tmp_path / "anomaly_detection_metrics_iid.csv"
    assert concise.exists()
    header = concise.read_text(encoding="utf-8").splitlines()[0]
    for field in ("loss", "accuracy", "precision", "recall", "f1_score"):
        assert field in header.split(",")
