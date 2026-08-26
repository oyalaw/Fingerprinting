from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_fingerprint.architecture_models import (
    candidate_feature_columns,
    load_bundle,
    predict_hierarchy,
    train_hierarchy_bundle,
)
from ai_fingerprint.proxy import DEFAULT_PROXY_CONFIG
from ai_fingerprint.server import ExperimentServer
from ai_fingerprint.traffic.analysis import (
    PacketRecord,
    extract_multiscale_feature_rows,
)


def _packet(index, t, size=1000, direction="up"):
    if direction == "up":
        src, dst = "10.0.0.2", "10.0.0.1"
    else:
        src, dst = "10.0.0.1", "10.0.0.2"
    return PacketRecord(
        index=index,
        timestamp_epoch=t,
        frame_length=size,
        src_ip=src,
        dst_ip=dst,
        src_port=50000 if direction == "up" else 5000,
        dst_port=5000 if direction == "up" else 50000,
        transport_protocol="TCP",
        tcp_flags_hex="0x10",
        tcp_syn=0,
        tcp_ack=1,
        tcp_fin=0,
        tcp_rst=0,
        retransmission=0,
        tls_record_lengths=(),
        direction=direction,
    )


def test_multiscale_feature_rows_include_online_scales():
    packets = [
        _packet(1, 1000.10, 1000, "down"),
        _packet(2, 1000.60, 1200, "down"),
        _packet(3, 1001.20, 900, "up"),
    ]
    rows = extract_multiscale_feature_rows(
        packets,
        "EXP_MULTI",
        window_sizes_sec=[0.5, 1.0, 2.0],
    )
    assert rows[0]["row_type"] == "overall"
    assert rows[0]["window_size_sec"] == 0.0

    scales = {
        float(row["window_size_sec"])
        for row in rows
        if row["row_type"] == "window"
    }
    assert scales == {0.5, 1.0, 2.0}


def _write_xy(tmp_path: Path):
    x = tmp_path / "fingerprinting_X_proxy.csv"
    y = tmp_path / "fingerprinting_Y_ground_truth.csv"

    x_fields = [
        "row_id",
        "experiment_id",
        "client_capture_id",
        "row_type",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "window_size_sec",
        "bytes_total",
        "packet_count_total",
        "packet_size_mean",
        "iat_sec_mean",
        "bytes_per_second",
    ]
    y_fields = [
        "row_id",
        "experiment_id",
        "family",
        "architecture",
        "variant",
    ]

    x_rows = []
    y_rows = []
    row_id = 0

    # Four independent runs per architecture, two samples per run.
    for architecture, variant, base in [
        ("resnet", "resnet101", 170_000_000),
        ("mobilenet", "mobilenetv2", 9_000_000),
    ]:
        for run in range(4):
            experiment_id = f"{architecture}_{run}"
            for client in range(2):
                row_id += 1
                total = base + run * 1000 + client * 100
                x_rows.append(
                    {
                        "row_id": row_id,
                        "experiment_id": experiment_id,
                        "client_capture_id": f"trace_{client}",
                        "row_type": "overall",
                        "window_index": -1,
                        "window_start_sec": 0,
                        "window_end_sec": 10,
                        "window_size_sec": 0,
                        "bytes_total": total,
                        "packet_count_total": total // 1400,
                        "packet_size_mean": 1300 + client,
                        "iat_sec_mean": (
                            0.0007
                            if architecture == "resnet"
                            else 0.0020
                        ),
                        "bytes_per_second": total / 10,
                    }
                )
                y_rows.append(
                    {
                        "row_id": row_id,
                        "experiment_id": experiment_id,
                        "family": "cnn",
                        "architecture": architecture,
                        "variant": variant,
                    }
                )

    with x.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=x_fields)
        writer.writeheader()
        writer.writerows(x_rows)
    with y.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=y_fields)
        writer.writeheader()
        writer.writerows(y_rows)
    return x, y


def test_final_hierarchy_model_learns_architecture(tmp_path):
    x, y = _write_xy(tmp_path)
    result = train_hierarchy_bundle(
        x,
        y,
        tmp_path / "models",
        mode="final",
        feature_mode="full",
    )
    bundle = load_bundle(result["bundle_path"])
    prediction = predict_hierarchy(
        bundle,
        {
            "bytes_total": 168_000_000,
            "packet_count_total": 120_000,
            "packet_size_mean": 1300,
            "iat_sec_mean": 0.0007,
            "bytes_per_second": 16_800_000,
        },
    )
    assert prediction["family"]["label"] == "cnn"
    assert prediction["architecture"]["label"] == "resnet"
    # With only one known variant under ResNet this is a constant candidate,
    # not evidence of variant-level discrimination.
    assert prediction["variant"]["label"] == "resnet101"
    assert prediction["variant"]["kind"] == "constant"


def test_size_normalized_mode_drops_absolute_footprint():
    columns = candidate_feature_columns(
        [
            "row_id",
            "experiment_id",
            "bytes_total",
            "packet_count_total",
            "bytes_per_second",
            "packet_size_mean",
            "iat_sec_mean",
        ],
        "size_normalized",
    )
    assert "bytes_total" not in columns
    assert "packet_count_total" not in columns
    assert "bytes_per_second" in columns
    assert "packet_size_mean" in columns


def test_proxy_defaults_enable_dual_inference_and_multiscale():
    capture = DEFAULT_PROXY_CONFIG["capture"]
    inference = DEFAULT_PROXY_CONFIG["architecture_inference"]
    assert capture["window_sizes_sec"] == [0.5, 1.0, 2.0, 5.0]
    assert inference["realtime_enabled"] is True
    assert inference["final_enabled"] is True
    assert set(inference["feature_modes"]) == {
        "full",
        "size_normalized",
    }


def test_server_rejects_federated_experiment_id_mismatch():
    server = ExperimentServer.__new__(ExperimentServer)
    server.config = {
        "experiment": {"experiment_id": "RUN_7"}
    }
    server._validate_federated_experiment_id(
        {"experiment_id": "RUN_7"}
    )

    try:
        server._validate_federated_experiment_id(
            {"experiment_id": "RUN_8"}
        )
    except RuntimeError as exc:
        assert "mismatch" in str(exc).lower()
    else:
        raise AssertionError("mismatched experiment ID was accepted")
