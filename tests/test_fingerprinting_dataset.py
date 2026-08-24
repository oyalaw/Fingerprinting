from __future__ import annotations

import csv
import json

import pytest

from ai_fingerprint.fingerprinting_dataset import (
    FingerprintingDataError,
    build_fingerprinting_dataset,
    load_xy,
    sanitize_packet_sequence,
)


def _write_proxy_features(path, experiment_id="EXP_001", extra=None):
    row = {
        "experiment_id": experiment_id,
        "row_type": "overall",
        "window_index": "-1",
        "window_start_sec": "0.0",
        "window_end_sec": "2.0",
        "packet_count_total": "10",
        "bytes_total": "4096",
        "bytes_up": "2048",
        "bytes_down": "2048",
        "iat_mean": "0.02",
    }
    if extra:
        row.update(extra)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _base_gt(experiment_id, role, device):
    return {
        "experiment_id": experiment_id,
        "role": role,
        "task": "training",
        "deployment": "federated",
        "framework": "pytorch",
        "runtime": "native",
        "family": "cnn",
        "architecture": "resnet",
        "variant": "resnet18",
        "application": "image_classification",
        "device": device,
        "dataset": "cifar10",
        "dataset_split": "train",
        "operating_system": (
            "ubuntu-client"
            if role == "client"
            else "ubuntu-server"
        ),
        "precision": "fp32",
        "batch_size": 4,
        "input_size": 32,
        "event": "experiment_start",
        "timestamp_utc": "2026-08-21T00:00:00+00:00",
    }


def _write_gt(path, record):
    path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )


def test_builder_physically_separates_proxy_x_and_ground_truth_y(tmp_path):
    features = tmp_path / "EXP_001_features.csv"
    client_gt = tmp_path / "EXP_001_client_ground_truth.jsonl"
    server_gt = tmp_path / "EXP_001_server_ground_truth.jsonl"

    _write_proxy_features(features)
    _write_gt(
        client_gt,
        _base_gt("EXP_001", "client", "jetson_orin_nano"),
    )
    _write_gt(
        server_gt,
        _base_gt("EXP_001", "server", "server_gpu"),
    )

    result = build_fingerprinting_dataset(
        proxy_feature_csvs=[features],
        ground_truth_jsonls=[client_gt, server_gt],
        output_dir=tmp_path / "dataset",
    )

    with open(result["X_proxy_csv"], newline="", encoding="utf-8") as handle:
        x_reader = csv.DictReader(handle)
        x_rows = list(x_reader)

    with open(result["Y_ground_truth_csv"], newline="", encoding="utf-8") as handle:
        y_reader = csv.DictReader(handle)
        y_rows = list(y_reader)

    assert len(x_rows) == 1
    assert len(y_rows) == 1

    # Ground-truth and resource fields cannot appear in X.
    for forbidden in {
        "task",
        "deployment",
        "framework",
        "runtime",
        "family",
        "architecture",
        "variant",
        "application",
        "device",
        "cpu_usage_percent",
        "gpu_usage_percent",
        "memory_usage_mb",
        "cpu_power_w",
        "cpu_energy_j",
    }:
        assert forbidden not in x_reader.fieldnames

    assert y_rows[0]["device"] == "jetson_orin_nano"
    assert y_rows[0]["family"] == "cnn"

    X, y, predictors = load_xy(
        result["X_proxy_csv"],
        result["Y_ground_truth_csv"],
        result["schema_json"],
        target="family",
    )
    assert y == ["cnn"]
    assert len(X) == 1
    assert "experiment_id" not in predictors
    assert "row_type" not in predictors
    assert "packet_count_total" in predictors


def test_builder_rejects_resource_or_ground_truth_columns_in_x(tmp_path):
    features = tmp_path / "EXP_001_features.csv"
    client_gt = tmp_path / "EXP_001_client_ground_truth.jsonl"

    _write_proxy_features(
        features,
        extra={
            "cpu_usage_percent": "12.5",
            "family": "cnn",
        },
    )
    _write_gt(
        client_gt,
        _base_gt("EXP_001", "client", "jetson_orin_nano"),
    )

    with pytest.raises(FingerprintingDataError):
        build_fingerprinting_dataset(
            proxy_feature_csvs=[features],
            ground_truth_jsonls=[client_gt],
            output_dir=tmp_path / "dataset",
        )


def test_builder_rejects_local_as_network_fingerprint_sample(tmp_path):
    features = tmp_path / "EXP_LOCAL_features.csv"
    client_gt = tmp_path / "EXP_LOCAL_client_ground_truth.jsonl"

    _write_proxy_features(features, experiment_id="EXP_LOCAL")
    record = _base_gt(
        "EXP_LOCAL",
        "client",
        "jetson_orin_nano",
    )
    record["deployment"] = "local"
    _write_gt(client_gt, record)

    with pytest.raises(FingerprintingDataError, match="labeled local"):
        build_fingerprinting_dataset(
            proxy_feature_csvs=[features],
            ground_truth_jsonls=[client_gt],
            output_dir=tmp_path / "dataset",
        )


def test_fingerprint_sequence_removes_endpoint_identity(tmp_path):
    raw = tmp_path / "EXP_001_packet_sequence.csv"
    safe = tmp_path / "EXP_001_fingerprint_sequence.csv"

    row = {
        "experiment_id": "EXP_001",
        "packet_index": "1",
        "timestamp_epoch": "1770000000.123",
        "relative_time_sec": "0.0",
        "frame_length": "1514",
        "direction": "up",
        "src_ip": "10.42.0.145",
        "dst_ip": "10.42.0.1",
        "src_port": "50123",
        "dst_port": "5000",
        "transport_protocol": "TCP",
        "tcp_flags_hex": "0x0018",
        "tcp_syn": "0",
        "tcp_ack": "1",
        "tcp_fin": "0",
        "tcp_rst": "0",
        "retransmission": "0",
        "tls_record_count": "1",
        "tls_record_lengths": "1448",
    }

    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    sanitize_packet_sequence(raw, safe)

    with safe.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 1
    for removed in {
        "timestamp_epoch",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
    }:
        assert removed not in reader.fieldnames

    assert rows[0]["frame_length"] == "1514"
    assert rows[0]["direction"] == "up"


def test_builder_joins_multi_client_proxy_alias_to_federated_client_id(tmp_path):
    exp = "EXP_MULTI"
    f1 = tmp_path / "EXP_MULTI__client_1_features.csv"
    f2 = tmp_path / "EXP_MULTI__client_2_features.csv"
    _write_proxy_features(
        f1,
        experiment_id=exp,
        extra={"client_capture_id": "client_1"},
    )
    _write_proxy_features(
        f2,
        experiment_id=exp,
        extra={"client_capture_id": "client_2"},
    )

    c1 = _base_gt(exp, "client", "jetson_orin_nano")
    c1["client_id"] = "client_1"
    c2 = _base_gt(exp, "client", "dell_desktop")
    c2["client_id"] = "client_2"
    s = _base_gt(exp, "server", "jetson_agx_orin")

    c1_path = tmp_path / "EXP_MULTI_client1_ground_truth.jsonl"
    c2_path = tmp_path / "EXP_MULTI_client2_ground_truth.jsonl"
    s_path = tmp_path / "EXP_MULTI_server_ground_truth.jsonl"
    _write_gt(c1_path, c1)
    _write_gt(c2_path, c2)
    _write_gt(s_path, s)

    result = build_fingerprinting_dataset(
        proxy_feature_csvs=[f1, f2],
        ground_truth_jsonls=[c1_path, c2_path, s_path],
        output_dir=tmp_path / "dataset",
    )

    with open(result["X_proxy_csv"], newline="", encoding="utf-8") as handle:
        x_reader = csv.DictReader(handle)
        x_rows = list(x_reader)
    with open(result["Y_ground_truth_csv"], newline="", encoding="utf-8") as handle:
        y_reader = csv.DictReader(handle)
        y_rows = list(y_reader)

    assert "client_capture_id" in x_reader.fieldnames
    assert "client_capture_id" not in json.loads(
        open(result["schema_json"], encoding="utf-8").read()
    )["predictor_columns"]
    assert [row["client_capture_id"] for row in x_rows] == [
        "client_1",
        "client_2",
    ]
    assert [row["device"] for row in y_rows] == [
        "jetson_orin_nano",
        "dell_desktop",
    ]


def test_builder_rejects_combined_features_when_multiple_clients_exist(tmp_path):
    exp = "EXP_MULTI"
    combined = tmp_path / "EXP_MULTI_features.csv"
    _write_proxy_features(combined, experiment_id=exp)

    c1 = _base_gt(exp, "client", "jetson_orin_nano")
    c1["client_id"] = "client_1"
    c2 = _base_gt(exp, "client", "dell_desktop")
    c2["client_id"] = "client_2"
    c1_path = tmp_path / "EXP_MULTI_client1_ground_truth.jsonl"
    c2_path = tmp_path / "EXP_MULTI_client2_ground_truth.jsonl"
    _write_gt(c1_path, c1)
    _write_gt(c2_path, c2)

    with pytest.raises(
        FingerprintingDataError,
        match="Use the per-client proxy feature files",
    ):
        build_fingerprinting_dataset(
            proxy_feature_csvs=[combined],
            ground_truth_jsonls=[c1_path, c2_path],
            output_dir=tmp_path / "dataset",
        )
