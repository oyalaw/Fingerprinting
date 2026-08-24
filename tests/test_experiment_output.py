from __future__ import annotations

from pathlib import Path

import pytest

from ai_fingerprint.experiment_output import (
    ExistingExperimentError,
    archive_existing_outputs,
    enforce_experiment_output_policy,
    find_existing_outputs,
)


def _config(tmp_path, role="server", policy="error"):
    return {
        "experiment": {
            "experiment_id": "RUN_001",
            "output_dir": str(tmp_path),
            "existing_output_policy": policy,
        },
        "node": {
            "role": role,
        },
    }


def test_role_specific_collision_detection(tmp_path):
    server_file = tmp_path / "RUN_001_server_ground_truth.jsonl"
    client_file = tmp_path / "RUN_001_client_ground_truth.jsonl"
    server_file.write_text("server\n", encoding="utf-8")
    client_file.write_text("client\n", encoding="utf-8")

    server_found = find_existing_outputs(
        _config(tmp_path, role="server"),
        role="server",
    )
    client_found = find_existing_outputs(
        _config(tmp_path, role="client"),
        role="client",
    )

    assert server_found == [server_file]
    assert client_found == [client_file]


def test_default_policy_refuses_existing_run(tmp_path):
    existing = tmp_path / "RUN_001_server_ground_truth.jsonl"
    existing.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ExistingExperimentError,
        match="Refusing to append or mix multiple runs",
    ):
        enforce_experiment_output_policy(
            _config(tmp_path, role="server", policy="error"),
            role="server",
        )

    assert existing.exists()


def test_archive_policy_moves_old_role_outputs(tmp_path):
    old_gt = tmp_path / "RUN_001_server_ground_truth.jsonl"
    old_csv = tmp_path / "RUN_001_server_resource.csv"
    unrelated_client = tmp_path / "RUN_001_client_ground_truth.jsonl"

    old_gt.write_text("old-gt\n", encoding="utf-8")
    old_csv.write_text("old-resource\n", encoding="utf-8")
    unrelated_client.write_text("client\n", encoding="utf-8")

    config = _config(tmp_path, role="server", policy="archive")
    enforce_experiment_output_policy(
        config,
        role="server",
    )

    assert not old_gt.exists()
    assert not old_csv.exists()
    assert unrelated_client.exists()
    assert find_existing_outputs(config, role="server") == []

    archived_gt = list(
        (tmp_path / "_archive" / "RUN_001" / "server").rglob(
            "RUN_001_server_ground_truth.jsonl"
        )
    )
    archived_csv = list(
        (tmp_path / "_archive" / "RUN_001" / "server").rglob(
            "RUN_001_server_resource.csv"
        )
    )
    assert len(archived_gt) == 1
    assert len(archived_csv) == 1
    assert archived_gt[0].read_text(encoding="utf-8") == "old-gt\n"


def test_proxy_collision_detection_is_separate(tmp_path):
    proxy_pcap = tmp_path / "RUN_001.pcapng"
    proxy_features = tmp_path / "RUN_001_features.csv"
    server_gt = tmp_path / "RUN_001_server_ground_truth.jsonl"

    proxy_pcap.write_bytes(b"pcap")
    proxy_features.write_text("x\n", encoding="utf-8")
    server_gt.write_text("{}\n", encoding="utf-8")

    config = {
        "experiment": {
            "experiment_id": "RUN_001",
            "output_dir": str(tmp_path),
            "existing_output_policy": "error",
        }
    }

    found = find_existing_outputs(config, role="proxy")
    assert proxy_pcap in found
    assert proxy_features in found
    assert server_gt not in found


def test_proxy_collision_detects_per_client_artifacts(tmp_path):
    per_client = tmp_path / "RUN_001__client_1_features.csv"
    per_client.write_text("x\n", encoding="utf-8")
    config = {
        "experiment": {
            "experiment_id": "RUN_001",
            "output_dir": str(tmp_path),
            "existing_output_policy": "error",
        }
    }
    found = find_existing_outputs(config, role="proxy")
    assert per_client in found


def test_federated_client_collision_is_scoped_to_client_id(tmp_path):
    c1 = tmp_path / "RUN_001_client_1_ground_truth.jsonl"
    c1.write_text("{}\n", encoding="utf-8")
    config = {
        "experiment": {
            "experiment_id": "RUN_001",
            "output_dir": str(tmp_path),
            "existing_output_policy": "error",
        },
        "node": {"role": "client"},
        "execution": {"deployment": "federated"},
        "federated": {"client_id": "client_2"},
    }
    assert find_existing_outputs(config, role="client") == []
