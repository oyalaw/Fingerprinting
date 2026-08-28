from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ai_fingerprint.config import DEFAULT_CONFIG, ConfigError, validate_config
from ai_fingerprint.dataset_manager import DatasetManager
from ai_fingerprint.experiment_integrity import RoundCheckpointManager
from ai_fingerprint.experiment_layout import write_role_status
from ai_fingerprint.training_metrics import classification_metrics


def _federated_config(tmp_path: Path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["experiment_id"] = "exp1"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["execution"].update({"task": "training", "deployment": "federated", "batch_size": 4})
    config["node"]["role"] = "server"
    config["federated"].update({"expected_clients": 3})
    return config


def test_fullscale_and_smoke_round_policy(tmp_path):
    full = _federated_config(tmp_path / "full")
    full["federated"].update({"mode": "full_scale", "rounds": 100})
    validate_config(full)
    full["federated"]["rounds"] = 99
    with pytest.raises(ConfigError):
        validate_config(full)

    smoke = _federated_config(tmp_path / "smoke")
    smoke["federated"].update({"mode": "smoke_test", "rounds": 10})
    validate_config(smoke)
    smoke["federated"]["rounds"] = 100
    with pytest.raises(ConfigError):
        validate_config(smoke)


def _anomaly_config(tmp_path: Path, client_index: int, role: str = "train"):
    config = _federated_config(tmp_path)
    config["node"]["role"] = "client"
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
        "type": "iid", "seed": 1234, "client_count": 3,
        "client_index": client_index, "client_id": f"client_{client_index+1}",
    })
    config["anomaly_detection"].update({
        "anomaly_labels": [9], "calibration_fraction": 0.10,
        "calibration_seed_offset": 73001, "data_role": role,
    })
    return config


def test_anomaly_calibration_is_disjoint_from_all_client_training(tmp_path):
    calibration = DatasetManager(_anomaly_config(tmp_path / "cal", 0, "calibration"))
    calibration_set = set(calibration._indices.tolist())
    assert calibration_set
    train_union: set[int] = set()
    for client_index in range(3):
        manager = DatasetManager(_anomaly_config(tmp_path / f"c{client_index}", client_index, "train"))
        indices = set(manager._indices.tolist())
        assert indices.isdisjoint(calibration_set)
        train_union.update(indices)
        assert 9 not in [manager._partition_label(i) for i in manager._indices]
    assert train_union.isdisjoint(calibration_set)
    assert 9 not in [calibration._partition_label(i) for i in calibration._indices]


def test_checkpoint_progress_every_round_but_model_archive_periodic(tmp_path):
    config = _federated_config(tmp_path)
    config["federated"]["rounds"] = 100
    config["checkpoint"].update({"enabled": True, "interval_rounds": 10, "retain_archives": 3})
    manager = RoundCheckpointManager(config)
    manager.start()
    params = [np.arange(4, dtype=np.float32)]
    manager.record_round(0, params, 3)
    progress = json.loads((tmp_path / "round_progress.json").read_text())
    assert progress["last_completed_round"] == 1
    assert not (tmp_path / "checkpoints" / "checkpoint_latest.npz").exists()
    manager.record_round(9, params, 3)
    assert (tmp_path / "checkpoints" / "checkpoint_latest.npz").exists()
    assert (tmp_path / "checkpoints" / "round_0010.npz").exists()


def test_classification_metrics_expose_unambiguous_macro_names():
    result = classification_metrics([0, 1, 2, 2], [0, 1, 1, 2])
    assert result["macro_precision"] == result["precision"]
    assert result["macro_recall"] == result["recall"]
    assert result["macro_f1"] == result["f1"]


def test_generic_complete_does_not_erase_integrity_failure(tmp_path):
    config = _federated_config(tmp_path)
    status_path = tmp_path / "experiment_status.json"
    status_path.write_text(json.dumps({"status": "METRICS_INCOMPLETE"}), encoding="utf-8")
    write_role_status(config, "COMPLETE")
    assert json.loads(status_path.read_text())["status"] == "METRICS_INCOMPLETE"
