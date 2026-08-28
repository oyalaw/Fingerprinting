from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from ai_fingerprint.client import ExperimentClient
from ai_fingerprint.config import DEFAULT_CONFIG, validate_config
from ai_fingerprint.fingerprinting_dataset import (
    build_fingerprinting_dataset,
)
from ai_fingerprint.resource_monitor import NvidiaSmiCollector
from ai_fingerprint.traffic.analysis import (
    PacketRecord,
    _export_packets_artifacts,
)


def _gt(exp, client_id, device):
    return {
        "experiment_id": exp,
        "role": "client",
        "client_id": client_id,
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
        "operating_system": "linux",
        "precision": "fp32",
        "batch_size": 1,
        "input_size": 32,
    }


def _write_gt(path, records):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )


def _write_feature(path, exp, capture_id):
    row = {
        "experiment_id": exp,
        "client_capture_id": capture_id,
        "row_type": "overall",
        "window_index": "-1",
        "window_start_sec": "0",
        "window_end_sec": "5",
        "trace_start_offset_sec": "1.25",
        "trace_end_offset_sec": "6.25",
        "window_start_global_sec": "1.25",
        "window_end_global_sec": "6.25",
        "packet_count_total": "10",
        "bytes_total": "10000",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_capture_id_map_corrects_reversed_manual_aliases(tmp_path):
    exp = "EXP_MAP"
    f1 = tmp_path / "trace_a_features.csv"
    f2 = tmp_path / "trace_b_features.csv"
    _write_feature(f1, exp, "client_001")
    _write_feature(f2, exp, "client_002")

    c1 = tmp_path / "c1_ground_truth.jsonl"
    c2 = tmp_path / "c2_ground_truth.jsonl"
    _write_gt(c1, [_gt(exp, "client_1", "jetson")])
    _write_gt(c2, [_gt(exp, "client_2", "dell")])

    result = build_fingerprinting_dataset(
        proxy_feature_csvs=[f1, f2],
        ground_truth_jsonls=[c1, c2],
        output_dir=tmp_path / "dataset",
        client_capture_id_map={
            (exp, "client_001"): "client_2",
            (exp, "client_002"): "client_1",
        },
    )

    with open(
        result["Y_ground_truth_csv"],
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["client_capture_id"] == "client_001"
    assert rows[0]["resolved_client_id"] == "client_2"
    assert rows[0]["device"] == "dell"
    assert rows[1]["resolved_client_id"] == "client_1"
    assert rows[1]["device"] == "jetson"

    schema = json.loads(
        Path(result["schema_json"]).read_text(encoding="utf-8")
    )
    predictors = set(schema["predictor_columns"])
    assert "trace_start_offset_sec" not in predictors
    assert "window_start_global_sec" not in predictors


def test_per_client_artifacts_preserve_global_trace_offset(tmp_path):
    packets = [
        PacketRecord(
            index=1,
            timestamp_epoch=1000.0,
            frame_length=100,
            src_ip="10.0.0.10",
            dst_ip="10.0.0.1",
            src_port=50000,
            dst_port=5000,
            transport_protocol="TCP",
            tcp_flags_hex="0x10",
            tcp_syn=0,
            tcp_ack=1,
            tcp_fin=0,
            tcp_rst=0,
            retransmission=0,
            tls_record_lengths=(),
            direction="up",
        ),
        PacketRecord(
            index=2,
            timestamp_epoch=1002.5,
            frame_length=120,
            src_ip="10.0.0.20",
            dst_ip="10.0.0.1",
            src_port=50001,
            dst_port=5000,
            transport_protocol="TCP",
            tcp_flags_hex="0x10",
            tcp_syn=0,
            tcp_ack=1,
            tcp_fin=0,
            tcp_rst=0,
            retransmission=0,
            tls_record_lengths=(),
            direction="up",
        ),
        PacketRecord(
            index=3,
            timestamp_epoch=1003.0,
            frame_length=140,
            src_ip="10.0.0.1",
            dst_ip="10.0.0.20",
            src_port=5000,
            dst_port=50001,
            transport_protocol="TCP",
            tcp_flags_hex="0x10",
            tcp_syn=0,
            tcp_ack=1,
            tcp_fin=0,
            tcp_rst=0,
            retransmission=0,
            tls_record_lengths=(),
            direction="down",
        ),
    ]

    result = _export_packets_artifacts(
        packets=packets,
        experiment_id="EXP_TIME",
        target_dir=tmp_path,
        source_capture={"parser": "unit"},
        client_ips=["10.0.0.10", "10.0.0.20"],
        client_aliases={
            "10.0.0.10": "trace_a",
            "10.0.0.20": "trace_b",
        },
        window_seconds=1.0,
    )

    b = result["per_client_artifacts"]["trace_b"]
    assert abs(b["trace_start_offset_sec"] - 2.5) < 1e-9

    with open(
        b["features_csv"],
        newline="",
        encoding="utf-8",
    ) as handle:
        first = next(csv.DictReader(handle))

    assert float(first["trace_start_offset_sec"]) == 2.5
    assert float(first["window_start_global_sec"]) == 2.5


def test_tls_server_config_allows_auto_generated_material():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["node"]["role"] = "server"
    config["execution"]["task"] = "training"
    config["execution"]["deployment"] = "federated"
    config["transport"]["kind"] = "tls"
    config["transport"]["certfile"] = None
    config["transport"]["keyfile"] = None
    config["transport"]["auto_generate_self_signed"] = True

    validate_config(config)


def test_unusable_nvidia_smi_is_disabled_after_probe(monkeypatch):
    import ai_fingerprint.resource_monitor as rm

    calls = []

    monkeypatch.setattr(
        rm.shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi"
        if name == "nvidia-smi"
        else None,
    )

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "No devices were found"

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return Completed()

    monkeypatch.setattr(rm.subprocess, "run", fake_run)

    collector = NvidiaSmiCollector(0)
    assert collector.available is False
    before = len(calls)
    sample = collector.sample()
    assert len(calls) == before
    assert sample["gpu_usage_percent"] is None


def test_federated_client_splits_upload_and_sync_wait(monkeypatch, tmp_path):
    import ai_fingerprint.client as client_mod

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["experiment_id"] = "EXP_PHASE"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["node"]["role"] = "client"
    config["node"]["host"] = "127.0.0.1"
    config["execution"]["task"] = "training"
    config["execution"]["deployment"] = "federated"
    config["federated"]["rounds"] = 1
    config["federated"]["local_epochs"] = 1
    config["federated"]["steps_per_epoch"] = 1
    config["federated"]["client_id"] = "client_1"

    events = []

    class Logger:
        def write(self, event, **fields):
            events.append((event, fields))

        def refresh_base(self, config):
            pass

    class Generator:
        def training_batch(self):
            return np.zeros((1, 1), dtype=np.float32), np.zeros(
                (1,), dtype=np.int64
            )

    class Workload:
        def set_parameters(self, arrays):
            pass

        def get_parameters(self):
            return [np.zeros((2,), dtype=np.float32)]

        def train_batch(self, inputs, targets):
            return {
                "loss": 1.0,
                "accuracy": 0.5,
                "learning_rate": 0.001,
            }

    class Sock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getsockname(self):
            return ("10.42.0.210", 45678)

        def shutdown(self, *args):
            pass

        def close(self):
            pass

    from ai_fingerprint.federated_policy import build_training_policy
    policy = build_training_policy(config)

    replies = iter(
        [
            (
                {
                    "status": "ok",
                    "training_policy": policy,
                },
                b"",
            ),
            (
                {
                    "status": "ok",
                    "done": False,
                    "round": 0,
                },
                client_mod.arrays_to_bytes(
                    [np.zeros((2,), dtype=np.float32)]
                ),
            ),
            (
                {
                    "status": "ok",
                    "done": True,
                    "next_round": 1,
                },
                b"",
            ),
        ]
    )

    obj = ExperimentClient.__new__(ExperimentClient)
    obj.config = config
    obj.logger = Logger()
    obj.generator = Generator()

    monkeypatch.setattr(
        client_mod,
        "build_workload",
        lambda config: Workload(),
    )
    monkeypatch.setattr(
        client_mod,
        "InputGenerator",
        lambda config: Generator(),
    )
    monkeypatch.setattr(
        obj,
        "_connect",
        lambda: Sock(),
    )
    monkeypatch.setattr(
        client_mod,
        "send_frame",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        client_mod,
        "recv_frame",
        lambda sock: next(replies),
    )
    monkeypatch.setattr(
        obj,
        "_close_remote",
        lambda sock: None,
    )

    obj._run_federated_training()

    phase_records = [
        fields
        for event, fields in events
        if event == "federated_phase"
    ]
    phases = [row["phase"] for row in phase_records]

    assert phases == [
        "Download",
        "Training",
        "Upload",
        "Idle",
    ]
    idle = phase_records[-1]
    assert idle["reason"] == "synchronous_round_wait"

    registrations = [
        fields
        for event, fields in events
        if event == "network_registration"
    ]
    assert registrations[0]["client_id"] == "client_1"
    assert registrations[0]["local_ip"] == "10.42.0.210"

    transactions = [
        fields
        for event, fields in events
        if event == "federated_upload_transaction"
    ]
    assert len(transactions) == 1
    assert "upload_transfer_time_ms" in transactions[0]
    assert "sync_wait_time_ms" in transactions[0]
