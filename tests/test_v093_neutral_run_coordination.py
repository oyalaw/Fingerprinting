from __future__ import annotations

from pathlib import Path

from ai_fingerprint.config import DEFAULT_CONFIG, deep_merge
from ai_fingerprint.experiment_coordination import (
    ActiveRunCoordinator,
    ensure_run_id,
    query_active_run,
)
from ai_fingerprint.experiment_layout import (
    apply_hierarchical_layout,
    apply_proxy_staging_layout,
)
from ai_fingerprint.proxy import DEFAULT_PROXY_CONFIG, validate_proxy_config


def _server_config(tmp_path: Path):
    cfg = deep_merge(DEFAULT_CONFIG, {})
    cfg["node"].update({"role": "server", "host": "127.0.0.1", "port": 8080})
    cfg["coordination"].update({"host": "127.0.0.1", "port": 0})
    cfg["execution"].update({"task": "training", "deployment": "federated"})
    cfg["ai"].update(
        {
            "family": "autoencoder",
            "architecture": "convolutional_autoencoder",
            "variant": "convolutional_autoencoder_6layer",
            "application": "anomaly_detection",
            "dataset": "fashion_mnist",
            "framework": "pytorch",
        }
    )
    apply_hierarchical_layout(
        cfg,
        root=tmp_path / "results",
        experiment_id="4",
        role_token="server",
    )
    return cfg


def test_server_generates_neutral_run_id_separate_from_hierarchy(tmp_path: Path):
    cfg = _server_config(tmp_path)
    run_id = ensure_run_id(cfg)
    assert run_id.startswith("run_")
    assert "autoencoder" not in run_id
    assert cfg["experiment"]["experiment_id"] == "exp4"
    assert cfg["experiment"]["storage_locator"].endswith(
        "autoencoder/convolutional_autoencoder/convolutional_autoencoder_6layer/"
        "anomaly_detection/fashion_mnist/pytorch/exp4"
    )


def test_proxy_discovers_only_neutral_run_id_from_server(tmp_path: Path):
    server_cfg = _server_config(tmp_path)
    coordinator = ActiveRunCoordinator(server_cfg)
    coordinator.start()
    try:
        info = query_active_run("127.0.0.1", coordinator.port)
    finally:
        coordinator.stop()
    assert info.run_id == server_cfg["experiment"]["run_id"]
    assert "autoencoder" not in info.run_id


def test_proxy_staging_layout_contains_no_hierarchy_labels(tmp_path: Path):
    cfg = {
        "experiment": dict(DEFAULT_PROXY_CONFIG["experiment"]),
        "coordination": dict(DEFAULT_PROXY_CONFIG["coordination"]),
        "proxy": dict(DEFAULT_PROXY_CONFIG["proxy"]),
        "capture": {
            **DEFAULT_PROXY_CONFIG["capture"],
            "interface": "wlo1",
        },
        "architecture_inference": dict(DEFAULT_PROXY_CONFIG["architecture_inference"]),
    }
    apply_proxy_staging_layout(
        cfg,
        staging_root=tmp_path / "staging",
        run_id="run_20260828t180000z_1234abcd",
    )
    output = Path(cfg["experiment"]["output_dir"])
    assert output == tmp_path / "staging" / "run_20260828t180000z_1234abcd" / "proxy"
    assert cfg["experiment"]["storage_locator"] is None
    assert "ai" not in cfg
    validate_proxy_config(cfg)


def test_proxy_defaults_use_out_of_band_coordinator_and_staging():
    assert DEFAULT_PROXY_CONFIG["coordination"]["server_host"] == "10.42.0.195"
    assert DEFAULT_PROXY_CONFIG["coordination"]["server_port"] == 8081
    assert DEFAULT_PROXY_CONFIG["experiment"]["results_root"] == "experiments/staging"
