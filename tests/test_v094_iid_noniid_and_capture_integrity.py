from __future__ import annotations

import copy
import csv
import json
import threading
import time

import numpy as np

from ai_fingerprint.config import DEFAULT_CONFIG
from ai_fingerprint.data_partition import make_partition_assignment
from ai_fingerprint.fingerprinting_dataset import _read_ground_truth_indices
from ai_fingerprint.server import ExperimentServer
from ai_fingerprint.traffic.analysis import PacketRecord, _export_packets_artifacts


def test_iid_partition_is_disjoint_complete_and_deterministic():
    labels = np.repeat(np.arange(10), 12)
    assignments = [
        make_partition_assignment(
            labels=labels,
            partition_type="iid",
            client_index=i,
            client_count=3,
            seed=42,
        )
        for i in range(3)
    ]
    sets = [set(item.indices.tolist()) for item in assignments]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(range(len(labels)))
    repeated = make_partition_assignment(
        labels=labels,
        partition_type="iid",
        client_index=1,
        client_count=3,
        seed=42,
    )
    assert np.array_equal(assignments[1].indices, repeated.indices)


def test_dirichlet_non_iid_is_disjoint_complete_and_label_skewed():
    labels = np.repeat(np.arange(10), 100)
    assignments = [
        make_partition_assignment(
            labels=labels,
            partition_type="non_iid",
            client_index=i,
            client_count=3,
            seed=43,
            alpha=0.1,
        )
        for i in range(3)
    ]
    sets = [set(item.indices.tolist()) for item in assignments]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(range(len(labels)))
    # Strong Dirichlet skew should make at least one client's dominant class
    # substantially larger than its uniform 10% baseline.
    dominant = []
    for assignment in assignments:
        counts = np.bincount(labels[assignment.indices], minlength=10)
        dominant.append(float(counts.max()) / max(int(counts.sum()), 1))
    assert max(dominant) > 0.25


def test_server_round_zero_barrier_waits_for_all_expected_clients(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["experiment_id"] = "exp1"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["node"]["role"] = "server"
    config["execution"]["task"] = "training"
    config["execution"]["deployment"] = "federated"
    config["federated"]["expected_clients"] = 3
    config["transport"]["kind"] = "tcp"
    server = ExperimentServer(config)

    released: list[str] = []

    def ready(client_id: str):
        server._wait_for_all_clients_ready(client_id)
        released.append(client_id)

    t1 = threading.Thread(target=ready, args=("client_1",))
    t2 = threading.Thread(target=ready, args=("client_2",))
    t1.start(); t2.start()
    time.sleep(0.05)
    assert released == []
    t3 = threading.Thread(target=ready, args=("client_3",))
    t3.start()
    for thread in (t1, t2, t3):
        thread.join(timeout=1.0)
    assert sorted(released) == ["client_1", "client_2", "client_3"]


def _packet(index, ts, src_ip, dst_ip, src_port, dst_port, direction):
    return PacketRecord(
        index=index,
        timestamp_epoch=ts,
        frame_length=100,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        transport_protocol="tcp",
        tcp_flags_hex="0x0018",
        tcp_syn=0,
        tcp_ack=1,
        tcp_fin=0,
        tcp_rst=0,
        retransmission=0,
        tls_record_lengths=(),
        direction=direction,
    )


def test_same_ip_different_source_ports_are_separate_proxy_traces(tmp_path):
    proxy_ip = "10.42.0.1"
    client_ip = "10.42.0.145"
    packets = [
        # stale/retry connection early
        _packet(1, 100.0, client_ip, proxy_ip, 57000, 8080, "up"),
        _packet(2, 101.0, proxy_ip, client_ip, 8080, 57000, "down"),
        # real FL connection much later
        _packet(3, 200.0, client_ip, proxy_ip, 51161, 8080, "up"),
        _packet(4, 201.0, proxy_ip, client_ip, 8080, 51161, "down"),
    ]
    result = _export_packets_artifacts(
        packets=packets,
        experiment_id="run_test",
        target_dir=tmp_path,
        source_capture={"source_kind": "test"},
        client_ips=[client_ip],
        client_connections=[
            {"client_ip": client_ip, "client_port": 57000, "alias": "trace_001"},
            {"client_ip": client_ip, "client_port": 51161, "alias": "trace_002"},
        ],
        proxy_ip=proxy_ip,
        proxy_port=8080,
        per_client_artifacts=True,
        window_sizes_sec=[0.5, 1.0, 2.0, 5.0],
    )
    traces = result["per_client_artifacts"]
    assert traces["trace_001"]["packet_count"] == 2
    assert traces["trace_002"]["packet_count"] == 2
    assert traces["trace_001"]["client_port"] == 57000
    assert traces["trace_002"]["client_port"] == 51161
    # Each trace's local duration is ~1s; the 100s gap between connections
    # must not contaminate either trace.
    assert traces["trace_001"]["trace_end_offset_sec"] - traces["trace_001"]["trace_start_offset_sec"] < 2
    assert traces["trace_002"]["trace_end_offset_sec"] - traces["trace_002"]["trace_start_offset_sec"] < 2


def test_ground_truth_uses_neutral_run_id_for_proxy_join(tmp_path):
    path = tmp_path / "exp1_client_1_ground_truth.jsonl"
    common = {
        "experiment_id": "exp1",
        "role": "client",
        "task": "training",
        "deployment": "federated",
        "framework": "pytorch",
        "runtime": "native",
        "family": "cnn",
        "architecture": "resnet",
        "variant": "resnet18",
        "application": "image_classification",
        "device": "dell_desktop",
        "client_id": "client_1",
    }
    before = dict(common, event="network_registration", run_id=None)
    after = dict(common, event="network_registration_confirmed", run_id="run_abc")
    path.write_text(json.dumps(before) + "\n" + json.dumps(after) + "\n", encoding="utf-8")
    _, client_labels = _read_ground_truth_indices([path])
    assert ("run_abc", "client_1") in client_labels
    assert ("exp1", "client_1") not in client_labels


def test_dataset_discovery_excludes_unmatched_stale_connection(tmp_path):
    from prepare_fingerprinting_dataset import discover_inputs

    run_id = "run_join"
    for capture_id in ("trace_001", "trace_002"):
        path = tmp_path / f"{run_id}__{capture_id}_features.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment_id", "client_capture_id", "row_type",
                    "window_index", "window_start_sec", "window_end_sec",
                    "bytes_total",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "experiment_id": run_id,
                "client_capture_id": capture_id,
                "row_type": "overall",
                "window_index": -1,
                "window_start_sec": 0,
                "window_end_sec": 1,
                "bytes_total": 100,
            })

    manifest = {
        "experiment_id": run_id,
        "outputs": {
            "per_client": {
                "trace_001": {"client_ip": "10.42.0.145", "client_port": 57000},
                "trace_002": {"client_ip": "10.42.0.145", "client_port": 51161},
            }
        },
    }
    (tmp_path / f"{run_id}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    gt = {
        "experiment_id": "exp1",
        "run_id": run_id,
        "event": "network_registration_confirmed",
        "role": "client",
        "client_id": "client_3",
        "local_ip": "10.42.0.145",
        "local_port": 51161,
        "task": "training",
        "deployment": "federated",
        "framework": "pytorch",
        "runtime": "native",
        "family": "autoencoder",
        "architecture": "convolutional_autoencoder",
        "variant": "convolutional_autoencoder_2layer",
        "application": "anomaly_detection",
        "device": "dell_laptop",
    }
    gt_path = tmp_path / "exp1_client_3_ground_truth.jsonl"
    gt_path.write_text(json.dumps(gt) + "\n", encoding="utf-8")

    proxy_features, ground_truth, client_map, diagnostics = discover_inputs(tmp_path)
    assert [path.name for path in proxy_features] == [
        f"{run_id}__trace_002_features.csv"
    ]
    assert ground_truth == [gt_path]
    assert client_map[(run_id, "trace_002")] == "client_3"
