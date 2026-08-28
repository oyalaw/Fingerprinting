from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ai_fingerprint import config as config_module
from ai_fingerprint.experiment_layout import (
    apply_hierarchical_layout,
    apply_proxy_locator_layout,
    branch_directory,
    locator_for,
    next_experiment_number,
    normalize_experiment_id,
)
from ai_fingerprint.federated_contract import build_model_contract, compact_contract
from ai_fingerprint.networking import interface_for_local_ip
from ai_fingerprint.proxy import DEFAULT_PROXY_CONFIG, validate_proxy_config


def _base_config():
    cfg = config_module.deep_merge(config_module.DEFAULT_CONFIG, {})
    cfg["node"]["role"] = "client"
    cfg["execution"]["task"] = "training"
    cfg["execution"]["deployment"] = "federated"
    cfg["ai"].update(
        {
            "framework": "pytorch",
            "runtime": "native",
            "family": "autoencoder",
            "architecture": "convolutional_autoencoder",
            "variant": "convolutional_autoencoder_6layer",
            "application": "anomaly_detection",
            "dataset": "fashion_mnist",
            "input_size": 32,
        }
    )
    cfg["federated"]["client_id"] = "client_2"
    return cfg


def test_hierarchical_layout_matches_requested_structure(tmp_path: Path):
    cfg = _base_config()
    apply_hierarchical_layout(
        cfg,
        root=tmp_path,
        experiment_id="12",
        role_token="client_2",
    )
    expected = (
        tmp_path
        / "autoencoder"
        / "convolutional_autoencoder"
        / "convolutional_autoencoder_6layer"
        / "anomaly_detection"
        / "fashion_mnist"
        / "pytorch"
        / "exp12"
        / "client_2"
    )
    assert Path(cfg["experiment"]["output_dir"]) == expected
    assert cfg["experiment"]["experiment_id"] == "exp12"
    assert locator_for(cfg).endswith(
        "autoencoder/convolutional_autoencoder/"
        "convolutional_autoencoder_6layer/anomaly_detection/"
        "fashion_mnist/pytorch/exp12"
    )


def test_next_experiment_uses_max_plus_one_not_first_gap(tmp_path: Path):
    cfg = _base_config()
    branch = branch_directory(tmp_path, cfg)
    (branch / "exp1").mkdir(parents=True)
    (branch / "exp2").mkdir()
    (branch / "exp4").mkdir()
    assert next_experiment_number(branch) == 5


def test_proxy_locator_places_proxy_under_same_run(tmp_path: Path):
    cfg = {
        **DEFAULT_PROXY_CONFIG,
        "experiment": dict(DEFAULT_PROXY_CONFIG["experiment"]),
        "proxy": dict(DEFAULT_PROXY_CONFIG["proxy"]),
        "capture": {
            **DEFAULT_PROXY_CONFIG["capture"],
            "client_ips": ["10.42.0.47", "10.42.0.210"],
            "interface": "wlo1",
        },
        "architecture_inference": dict(DEFAULT_PROXY_CONFIG["architecture_inference"]),
    }
    locator = (
        "autoencoder/convolutional_autoencoder/"
        "convolutional_autoencoder_6layer/anomaly_detection/"
        "fashion_mnist/pytorch/exp12"
    )
    apply_proxy_locator_layout(cfg, root=tmp_path, locator=locator)
    assert Path(cfg["experiment"]["output_dir"]) == tmp_path / locator / "proxy"
    assert cfg["experiment"]["experiment_id"] == "exp12"
    validate_proxy_config(cfg)


def test_proxy_defaults_match_testbed_addresses():
    assert DEFAULT_PROXY_CONFIG["proxy"]["listen_host"] == "10.42.0.1"
    assert DEFAULT_PROXY_CONFIG["proxy"]["listen_port"] == 8080
    assert DEFAULT_PROXY_CONFIG["proxy"]["upstream_host"] == "10.42.0.195"
    assert DEFAULT_PROXY_CONFIG["proxy"]["upstream_port"] == 8080
    assert DEFAULT_PROXY_CONFIG["capture"]["window_sizes_sec"] == [0.5, 1.0, 2.0, 5.0]


def test_model_contract_detects_tensor_contract_change():
    cfg = _base_config()
    a = build_model_contract(
        cfg,
        [np.zeros((4, 4), dtype=np.float32), np.zeros((4,), dtype=np.float32)],
    )
    b = build_model_contract(
        cfg,
        [
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4,), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
        ],
    )
    assert compact_contract(a)["contract_id"] != compact_contract(b)["contract_id"]
    assert a["tensor_count"] == 2
    assert b["tensor_count"] == 3


def test_normalize_experiment_id_accepts_integer_or_expn():
    assert normalize_experiment_id("12") == "exp12"
    assert normalize_experiment_id("exp12") == "exp12"
    with pytest.raises(ValueError):
        normalize_experiment_id("auto")


def test_model_contract_digest_is_run_salted_and_compact_has_no_labels():
    cfg1 = _base_config()
    cfg1["experiment"]["experiment_id"] = "exp1"
    cfg2 = _base_config()
    cfg2["experiment"]["experiment_id"] = "exp2"
    arrays = [np.zeros((2, 2), dtype=np.float32)]
    c1 = build_model_contract(cfg1, arrays)
    c2 = build_model_contract(cfg2, arrays)
    assert c1["contract_id"] != c2["contract_id"]
    compact = compact_contract(c1)
    assert set(compact) == {"contract_version", "contract_id", "tensor_count"}
