from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np


VALID_STATUSES = {
    "RUNNING",
    "COMPLETED",
    "PARTIAL",
    "FAILED",
    "CAPTURE_INCOMPLETE",
    "METRICS_INCOMPLETE",
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_preflight(path: str | Path, minimum_free_gb: float) -> Dict[str, Any]:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    gib = float(1024 ** 3)
    free_gb = usage.free / gib
    minimum = max(float(minimum_free_gb), 0.0)
    return {
        "path": str(target.resolve()),
        "total_gb": usage.total / gib,
        "used_gb": usage.used / gib,
        "free_gb": free_gb,
        "minimum_free_gb": minimum,
        "passed": free_gb >= minimum,
        "timestamp_utc": utc_now_iso(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def reproducibility_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    execution = config.get("execution", {}) or {}
    partition = config.get("data", {}).get("partition", {}) or {}
    packages = {
        name: _package_version(name)
        for name in (
            "numpy", "torch", "torchvision", "tensorflow", "flwr", "scikit-learn",
            "xgboost", "pandas", "pyyaml", "psutil",
        )
    }
    packages = {key: value for key, value in packages.items() if value is not None}

    cuda_version = None
    cudnn_version = None
    gpu_name = None
    if str(config.get("ai", {}).get("framework", "")).lower() == "pytorch":
        try:
            import torch
            cuda_version = getattr(torch.version, "cuda", None)
            if torch.backends.cudnn.is_available():
                cudnn_version = torch.backends.cudnn.version()
            if torch.cuda.is_available():
                gpu_index = int(config.get("resource_monitor", {}).get("gpu_index", 0))
                gpu_name = torch.cuda.get_device_name(gpu_index)
        except Exception:
            pass

    seed = int(execution.get("seed", 42))
    partition_seed = partition.get("seed")
    return {
        "timestamp_utc": utc_now_iso(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device_label": config.get("device", {}).get("label"),
        "configured_operating_system": config.get("device", {}).get("operating_system"),
        "framework": config.get("ai", {}).get("framework"),
        "runtime": config.get("ai", {}).get("runtime"),
        "package_versions": packages,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_name": gpu_name,
        "seeds": {
            "execution_seed": seed,
            "model_init_seed": seed,
            "shuffle_seed": seed,
            "partition_seed": int(partition_seed) if partition_seed is not None else None,
            "anomaly_calibration_seed": seed + int(config.get("anomaly_detection", {}).get("calibration_seed_offset", 73001)),
        },
    }


class RoundCheckpointManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cfg = config.get("checkpoint", {}) or {}
        self.enabled = bool(checkpoint_cfg.get("enabled", True))
        self.interval = max(int(checkpoint_cfg.get("interval_rounds", 10)), 1)
        self.retain = max(int(checkpoint_cfg.get("retain_archives", 3)), 1)
        self.target_rounds = int(config.get("federated", {}).get("rounds", 0))
        self.expected_clients = int(config.get("federated", {}).get("expected_clients", 0))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.progress_path = self.output_dir / "round_progress.json"
        self.status_path = self.output_dir / "experiment_status.json"

    def start(self) -> None:
        self._write_status("RUNNING", last_completed_round=0)
        atomic_write_json(self.progress_path, {
            "status": "RUNNING",
            "current_round": 1 if self.target_rounds else 0,
            "target_rounds": self.target_rounds,
            "clients_received": 0,
            "clients_expected": self.expected_clients,
            "last_completed_round": 0,
            "last_completed_round_index": None,
            "timestamp_utc": utc_now_iso(),
        })

    def record_round(self, round_index: int, parameters: Sequence[np.ndarray], clients_received: int) -> None:
        completed_number = int(round_index) + 1
        next_number = min(completed_number + 1, self.target_rounds)
        if self.enabled and (
            completed_number % self.interval == 0
            or completed_number == self.target_rounds
        ):
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._save_npz(self.checkpoint_dir / "checkpoint_latest.npz", parameters)
            atomic_write_json(self.checkpoint_dir / "checkpoint_latest.json", {
                "completed_round": completed_number,
                "round_index": int(round_index),
                "target_rounds": self.target_rounds,
                "checkpoint_interval_rounds": self.interval,
                "timestamp_utc": utc_now_iso(),
            })
            archive = self.checkpoint_dir / f"round_{completed_number:04d}.npz"
            self._save_npz(archive, parameters)
            self._prune_archives()

        atomic_write_json(self.progress_path, {
            "status": "RUNNING" if completed_number < self.target_rounds else "COMPLETED",
            "current_round": next_number,
            "target_rounds": self.target_rounds,
            "clients_received": int(clients_received),
            "clients_expected": self.expected_clients,
            "last_completed_round": completed_number,
            "last_completed_round_index": int(round_index),
            "timestamp_utc": utc_now_iso(),
        })

    def complete(self, *, metrics_complete: bool = True) -> None:
        status = "COMPLETED" if metrics_complete else "METRICS_INCOMPLETE"
        self._write_status(status, last_completed_round=self.target_rounds)

    def fail(self, error: str, last_completed_round: int) -> None:
        status = "PARTIAL" if int(last_completed_round) > 0 else "FAILED"
        self._write_status(status, error=error, last_completed_round=int(last_completed_round))

    def _write_status(self, status: str, **extra: Any) -> None:
        normalized = str(status).upper()
        if normalized not in VALID_STATUSES:
            raise ValueError(f"Unsupported experiment status: {status}")
        payload = {
            "experiment_id": self.config.get("experiment", {}).get("experiment_id"),
            "run_id": self.config.get("experiment", {}).get("run_id"),
            "role": self.config.get("node", {}).get("role"),
            "status": normalized,
            "target_rounds": self.target_rounds,
            "timestamp_utc": utc_now_iso(),
        }
        payload.update(extra)
        atomic_write_json(self.status_path, payload)

    @staticmethod
    def _save_npz(path: Path, parameters: Sequence[np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp.npz")
        np.savez_compressed(tmp, *[np.asarray(value) for value in parameters])
        os.replace(tmp, path)

    def _prune_archives(self) -> None:
        archives = sorted(self.checkpoint_dir.glob("round_*.npz"))
        for old in archives[:-self.retain]:
            try:
                old.unlink()
            except OSError:
                pass
