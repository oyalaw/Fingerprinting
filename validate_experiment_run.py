#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ai_fingerprint.experiment_integrity import atomic_write_json, utc_now_iso


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def validate_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    server_dir = run_dir / "server"
    proxy_dir = run_dir / "proxy"
    server_status = _json(server_dir / "experiment_status.json")
    progress = _json(server_dir / "round_progress.json")
    server_config = {}
    config_path = server_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            server_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            server_config = {}

    target_rounds = int(
        progress.get("target_rounds")
        or server_config.get("federated", {}).get("rounds", 0)
        or 0
    )
    expected_clients = int(
        server_config.get("federated", {}).get("expected_clients", 0)
        or progress.get("clients_expected", 0)
        or 0
    )
    round_rows = _csv_rows(server_dir / "round_metrics.csv")
    client_update_rows = _csv_rows(server_dir / "client_update_metrics.csv")
    last_completed = int(progress.get("last_completed_round", 0) or 0)

    proxy_status = _json(proxy_dir / "experiment_status.json")
    capture_manifest = _json(proxy_dir / "capture_chunks_manifest.json")
    proxy_present = proxy_dir.exists()
    capture_required = proxy_present

    client_dirs = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and path.name.lower().startswith("client")
    ) if run_dir.exists() else []
    client_checks = []
    for client_dir in client_dirs:
        status = _json(client_dir / "experiment_status.json")
        rows = _csv_rows(client_dir / "round_metrics.csv")
        client_checks.append({
            "client": client_dir.name,
            "status": status.get("status"),
            "round_metric_rows": rows,
            "complete": str(status.get("status", "")).upper() == "COMPLETED"
            and (target_rounds <= 0 or rows >= target_rounds),
        })

    checks = {
        "server_status_completed": str(server_status.get("status", "")).upper() == "COMPLETED",
        "server_rounds_completed": target_rounds > 0 and last_completed >= target_rounds,
        "server_round_metrics_complete": target_rounds > 0 and round_rows >= target_rounds,
        "server_client_update_metrics_complete": (
            target_rounds > 0
            and expected_clients > 0
            and client_update_rows >= target_rounds * expected_clients
        ),
        "clients_complete": all(item["complete"] for item in client_checks) if client_checks else True,
        "proxy_status_completed": (
            not capture_required
            or str(proxy_status.get("status", "")).upper() == "COMPLETED"
        ),
        "capture_complete": (
            not capture_required
            or (
                bool(capture_manifest.get("capture_complete", False))
                and int(capture_manifest.get("pcap_chunks", 0) or 0) > 0
                and int(capture_manifest.get("total_capture_bytes", 0) or 0) > 0
                and int(capture_manifest.get("total_packets", 0) or 0) > 0
            )
        ),
    }

    if not checks["server_rounds_completed"]:
        status = "PARTIAL" if last_completed > 0 else "FAILED"
    elif not checks["server_round_metrics_complete"] or not checks["server_client_update_metrics_complete"] or not checks["clients_complete"]:
        status = "METRICS_INCOMPLETE"
    elif not checks["proxy_status_completed"] or not checks["capture_complete"]:
        status = "CAPTURE_INCOMPLETE"
    elif all(checks.values()):
        status = "COMPLETED"
    else:
        status = "FAILED"

    result = {
        "status": status,
        "run_dir": str(run_dir),
        "target_rounds": target_rounds,
        "last_completed_round": last_completed,
        "expected_clients": expected_clients,
        "server_round_metric_rows": round_rows,
        "server_client_update_metric_rows": client_update_rows,
        "pcap_chunks": int(capture_manifest.get("pcap_chunks", 0) or 0),
        "total_capture_bytes": int(capture_manifest.get("total_capture_bytes", 0) or 0),
        "total_packets": int(capture_manifest.get("total_packets", 0) or 0),
        "checks": checks,
        "clients": client_checks,
        "timestamp_utc": utc_now_iso(),
    }
    atomic_write_json(run_dir / "experiment_status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one collected hierarchical fingerprinting expN run."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = validate_run(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
