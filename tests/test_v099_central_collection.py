from __future__ import annotations

import json
import threading
from pathlib import Path

import yaml

from ai_fingerprint.result_collection import (
    _CollectorHTTPServer,
    _CollectorHandler,
    auto_upload_result_copy,
    build_collection_archive,
    upload_result_copy,
    validate_collected_run,
)


def _client_config(output: Path, port: int, client_id: str = "client_1"):
    return {
        "experiment": {
            "experiment_id": "exp1",
            "run_id": "run_test_001",
            "output_dir": str(output),
            "storage_locator": "autoencoder/dense/dense_2layer/anomaly_detection/fashion_mnist/pytorch/exp1",
        },
        "node": {"role": "client"},
        "execution": {"deployment": "federated"},
        "federated": {"client_id": client_id},
        "result_collection": {
            "enabled": True,
            "collector_host": "127.0.0.1",
            "collector_port": port,
            "retry_attempts": 1,
            "timeout_sec": 3,
            "max_archive_mb": 32,
        },
    }


def _server_config(output: Path, port: int, expected_clients: int = 1):
    return {
        "experiment": {
            "experiment_id": "exp1",
            "run_id": "run_test_001",
            "output_dir": str(output),
        },
        "node": {"role": "server"},
        "execution": {"deployment": "federated"},
        "federated": {"expected_clients": expected_clients},
        "result_collection": {
            "enabled": True,
            "collector_host": "127.0.0.1",
            "collector_port": port,
            "retry_attempts": 1,
            "timeout_sec": 3,
            "max_archive_mb": 32,
        },
    }


def _proxy_config(output: Path, port: int):
    return {
        "experiment": {
            "experiment_id": "run_test_001",
            "run_id": "run_test_001",
            "output_dir": str(output),
        },
        "proxy": {"listen_host": "127.0.0.1"},
        "result_collection": {
            "enabled": True,
            "collector_host": "127.0.0.1",
            "collector_port": port,
            "retry_attempts": 1,
            "timeout_sec": 3,
            "max_archive_mb": 32,
        },
    }


def _start_collector(root: Path):
    server = _CollectorHTTPServer(
        ("127.0.0.1", 0),
        _CollectorHandler,
        root=root,
        shared_token="",
        max_upload_bytes=32 * 1024 * 1024,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_address[1]


def _status(output: Path, role: str):
    (output / "experiment_status.json").write_text(
        json.dumps(
            {
                "run_id": "run_test_001",
                "experiment_id": "exp1",
                "role": role,
                "status": "COMPLETED",
            }
        ),
        encoding="utf-8",
    )
    (output / "role_manifest.json").write_text(
        json.dumps({"run_id": "run_test_001", "role": role}),
        encoding="utf-8",
    )


def test_collection_archive_is_copy_only_and_excludes_raw_artifacts(tmp_path: Path):
    output = tmp_path / "client"
    output.mkdir()
    (output / "round_metrics.csv").write_text("round,loss\n1,0.5\n", encoding="utf-8")
    (output / "ground_truth.jsonl").write_text('{"run_id":"run_test_001"}\n', encoding="utf-8")
    (output / "pcap").mkdir()
    (output / "pcap" / "capture.pcapng").write_bytes(b"raw pcap")
    (output / "checkpoints").mkdir()
    (output / "checkpoints" / "round_0100.npz").write_bytes(b"weights")

    config = _client_config(output, 8090)
    archive, manifest = build_collection_archive(config, temp_dir=tmp_path)
    try:
        paths = {entry["path"] for entry in manifest["files"]}
        assert "round_metrics.csv" in paths
        assert "ground_truth.jsonl" in paths
        assert not any("pcap" in value for value in paths)
        assert not any("checkpoints" in value for value in paths)
        assert (output / "round_metrics.csv").exists()
        assert (output / "pcap" / "capture.pcapng").exists()
        assert manifest["local_result_retained"] is True
    finally:
        archive.unlink(missing_ok=True)


def test_upload_is_verified_idempotent_and_keeps_local_results(tmp_path: Path):
    central = tmp_path / "central"
    server, thread, port = _start_collector(central)
    try:
        output = tmp_path / "client"
        output.mkdir()
        _status(output, "client")
        (output / "run_test_001_client_1_ground_truth.jsonl").write_text(
            '{"run_id":"run_test_001","experiment_id":"exp1","client_id":"client_1"}\n',
            encoding="utf-8",
        )
        config = _client_config(output, port)

        first = upload_result_copy(config)
        second = upload_result_copy(config)
        assert first["status"] == "VERIFIED"
        assert second["status"] == "VERIFIED"
        assert first["content_sha256"] == second["content_sha256"]
        assert (output / "run_test_001_client_1_ground_truth.jsonl").exists()
        assert (output / "collection_receipt.json").exists()
        collected = central / "run_test_001" / "client_1"
        assert collected.exists()
        assert (collected / "run_test_001_client_1_ground_truth.jsonl").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_collection_failure_becomes_pending_without_touching_experiment_status(tmp_path: Path):
    output = tmp_path / "client"
    output.mkdir()
    _status(output, "client")
    (output / "metrics.csv").write_text("round,loss\n1,0.1\n", encoding="utf-8")
    config = _client_config(output, 65500)
    result = auto_upload_result_copy(config)
    assert result["status"] == "PENDING"
    experiment_status = json.loads((output / "experiment_status.json").read_text())
    collection_status = json.loads((output / "collection_status.json").read_text())
    assert experiment_status["status"] == "COMPLETED"
    assert collection_status["status"] == "PENDING"
    assert collection_status["local_results_retained"] is True


def test_central_validator_requires_server_proxy_and_expected_clients(tmp_path: Path):
    central = tmp_path / "central"
    httpd, thread, port = _start_collector(central)
    try:
        # Server
        server_out = tmp_path / "server"
        server_out.mkdir()
        _status(server_out, "server")
        (server_out / "config.yaml").write_text(
            yaml.safe_dump({"federated": {"expected_clients": 2}}), encoding="utf-8"
        )
        upload_result_copy(_server_config(server_out, port, expected_clients=2))

        # Client 1
        c1 = tmp_path / "c1"
        c1.mkdir()
        _status(c1, "client")
        (c1 / "run_test_001_client_1_ground_truth.jsonl").write_text(
            '{"run_id":"run_test_001","experiment_id":"exp1","client_id":"client_1"}\n',
            encoding="utf-8",
        )
        upload_result_copy(_client_config(c1, port, "client_1"))

        partial = validate_collected_run(central / "run_test_001")
        assert partial["status"] == "PARTIAL"
        assert any("missing proxy" in issue for issue in partial["issues"])
        assert any("expected 2 clients" in issue for issue in partial["issues"])

        # Client 2
        c2 = tmp_path / "c2"
        c2.mkdir()
        _status(c2, "client")
        (c2 / "run_test_001_client_2_ground_truth.jsonl").write_text(
            '{"run_id":"run_test_001","experiment_id":"exp1","client_id":"client_2"}\n',
            encoding="utf-8",
        )
        upload_result_copy(_client_config(c2, port, "client_2"))

        # Proxy; raw PCAP intentionally stays local and is not required centrally.
        proxy_out = tmp_path / "proxy"
        proxy_out.mkdir()
        _status(proxy_out, "proxy")
        (proxy_out / "run_test_001_features.csv").write_text(
            "experiment_id,row_type,packet_count_total,bytes_total\n"
            "run_test_001,final,100,10000\n",
            encoding="utf-8",
        )
        (proxy_out / "pcap").mkdir()
        (proxy_out / "pcap" / "capture.pcapng").write_bytes(b"large trace stays local")
        upload_result_copy(_proxy_config(proxy_out, port))

        valid = validate_collected_run(central / "run_test_001")
        assert valid["status"] == "VALID", valid["issues"]
        assert valid["collected_clients"] == ["client_1", "client_2"]
        assert not (central / "run_test_001" / "proxy" / "pcap" / "capture.pcapng").exists()
        assert (proxy_out / "pcap" / "capture.pcapng").exists()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
