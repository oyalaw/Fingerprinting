from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import psutil

from .metadata import output_role_token


RESOURCE_SCHEMA_VERSION = "1.0"

RESOURCE_FIELDS = [
    "experiment_id",
    "timestamp_utc",
    "timestamp_monotonic_ns",
    "relative_time_sec",
    "sample_index",
    "role",
    "device",
    "sample_interval_ms",
    "telemetry_source",
    "network_interface",
    "bytes_sent",
    "bytes_received",
    "bytes_sent_total",
    "bytes_received_total",
    "bytes_sent_delta",
    "bytes_received_delta",
    "cpu_usage_percent",
    "process_cpu_usage_percent_raw",
    "system_cpu_usage_percent",
    "memory_usage_mb",
    "memory_usage_percent",
    "system_memory_usage_percent",
    "gpu_usage_percent",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "cpu_power_w",
    "gpu_power_w",
    "system_power_w",
    "cpu_energy_j",
    "gpu_energy_j",
    "system_energy_j",
]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _read_number(path: Path) -> Optional[float]:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


class NetworkCounter:
    def __init__(self, interface: Optional[str]) -> None:
        self.requested_interface = interface
        self.interface = interface
        self._baseline_sent = 0
        self._baseline_recv = 0
        self._last_sent = 0
        self._last_recv = 0
        self._initialize()

    def _counters(self):
        if self.interface:
            pernic = psutil.net_io_counters(pernic=True)
            if self.interface not in pernic:
                available = ", ".join(sorted(pernic))
                raise RuntimeError(
                    f"Network interface {self.interface!r} was not found. "
                    f"Available interfaces: {available}"
                )
            return pernic[self.interface]
        return psutil.net_io_counters(pernic=False)

    def _initialize(self) -> None:
        counters = self._counters()
        self._baseline_sent = int(counters.bytes_sent)
        self._baseline_recv = int(counters.bytes_recv)
        self._last_sent = self._baseline_sent
        self._last_recv = self._baseline_recv

    def sample(self) -> Dict[str, int | str]:
        counters = self._counters()
        sent_total = int(counters.bytes_sent)
        recv_total = int(counters.bytes_recv)

        sent_delta = max(0, sent_total - self._last_sent)
        recv_delta = max(0, recv_total - self._last_recv)
        sent_experiment = max(0, sent_total - self._baseline_sent)
        recv_experiment = max(0, recv_total - self._baseline_recv)

        self._last_sent = sent_total
        self._last_recv = recv_total

        return {
            "network_interface": self.interface or "all",
            "bytes_sent": sent_experiment,
            "bytes_received": recv_experiment,
            "bytes_sent_total": sent_total,
            "bytes_received_total": recv_total,
            "bytes_sent_delta": sent_delta,
            "bytes_received_delta": recv_delta,
        }


class RaplCollector:
    """
    Intel/AMD package energy collector through Linux powercap sysfs.

    cpu_energy_j is reported relative to monitor start.
    cpu_power_w is derived from successive energy counter readings.
    """

    def __init__(self) -> None:
        self.energy_files = self._discover_energy_files()
        self._baseline_uj: Optional[float] = None
        self._last_uj: Optional[float] = None
        self._last_time: Optional[float] = None

        if self.energy_files:
            total = self._read_total_uj()
            self._baseline_uj = total
            self._last_uj = total
            self._last_time = time.monotonic()

    @staticmethod
    def _discover_energy_files() -> List[Path]:
        roots = [
            Path("/sys/class/powercap"),
            Path("/sys/devices/virtual/powercap"),
        ]
        candidates: List[Path] = []
        seen = set()

        for root in roots:
            if not root.exists():
                continue
            for path in root.glob("intel-rapl:*"):
                if ":" in path.name[len("intel-rapl:"):]:
                    # Avoid summing child domains into the package total.
                    continue
                energy = path / "energy_uj"
                if energy.exists():
                    resolved = str(energy.resolve())
                    if resolved not in seen:
                        candidates.append(energy)
                        seen.add(resolved)

            for path in root.glob("amd-rapl:*"):
                if ":" in path.name[len("amd-rapl:"):]:
                    continue
                energy = path / "energy_uj"
                if energy.exists():
                    resolved = str(energy.resolve())
                    if resolved not in seen:
                        candidates.append(energy)
                        seen.add(resolved)

        return candidates

    def _read_total_uj(self) -> Optional[float]:
        values = [_read_number(path) for path in self.energy_files]
        values = [value for value in values if value is not None]
        if not values:
            return None
        return float(sum(values))

    def sample(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.energy_files:
            return None, None

        current_uj = self._read_total_uj()
        now = time.monotonic()
        if current_uj is None:
            return None, None

        energy_j = None
        power_w = None

        if self._baseline_uj is not None:
            delta_from_start = current_uj - self._baseline_uj
            if delta_from_start >= 0:
                energy_j = delta_from_start / 1_000_000.0

        if self._last_uj is not None and self._last_time is not None:
            delta_uj = current_uj - self._last_uj
            delta_t = now - self._last_time
            if delta_uj >= 0 and delta_t > 0:
                power_w = (delta_uj / 1_000_000.0) / delta_t

        self._last_uj = current_uj
        self._last_time = now
        return power_w, energy_j

    @property
    def available(self) -> bool:
        return bool(self.energy_files)


class NvidiaSmiCollector:
    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = int(gpu_index)
        self.executable = shutil.which("nvidia-smi")

    @property
    def available(self) -> bool:
        return self.executable is not None

    def sample(self) -> Dict[str, Optional[float]]:
        empty = {
            "gpu_usage_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_power_w": None,
        }
        if not self.executable:
            return empty

        query = (
            "utilization.gpu,memory.used,memory.total,power.draw"
        )
        command = [
            self.executable,
            f"--id={self.gpu_index}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except Exception:
            return empty

        if completed.returncode != 0 or not completed.stdout.strip():
            return empty

        first_line = completed.stdout.strip().splitlines()[0]
        parts = [piece.strip() for piece in first_line.split(",")]
        if len(parts) < 4:
            return empty

        return {
            "gpu_usage_percent": _safe_float(parts[0]),
            "gpu_memory_used_mb": _safe_float(parts[1]),
            "gpu_memory_total_mb": _safe_float(parts[2]),
            "gpu_power_w": _safe_float(parts[3]),
        }


class JetsonCollector:
    """
    Best effort Jetson collector.

    GPU utilization is read from common devfreq sysfs nodes.
    System input power is read from INA3221/hwmon rails when a total-input
    label such as VDD_IN, POM_5V_IN, or VIN_SYS_5V0 is exposed.
    """

    TOTAL_POWER_LABELS = {
        "VDD_IN",
        "POM_5V_IN",
        "VIN_SYS_5V0",
        "VDD_SYS_IN",
    }

    CPU_POWER_LABELS = {
        "VDD_CPU",
        "VDD_SYS_CPU",
        "VDD_CPU_GPU_CV",
    }

    GPU_POWER_LABELS = {
        "VDD_GPU",
        "VDD_SYS_GPU",
        "VDD_CPU_GPU_CV",
    }

    def __init__(self) -> None:
        self.gpu_load_paths = self._discover_gpu_load_paths()
        self.power_candidates = self._discover_power_candidates()

    @staticmethod
    def _discover_gpu_load_paths() -> List[Path]:
        patterns = [
            "/sys/devices/*gpu*/load",
            "/sys/devices/platform/*gpu*/load",
            "/sys/class/devfreq/*gpu*/load",
        ]
        paths: List[Path] = []
        for pattern in patterns:
            paths.extend(Path("/").glob(pattern.lstrip("/")))
        return [path for path in paths if path.exists()]

    @classmethod
    def _discover_power_candidates(cls) -> List[Tuple[str, Path]]:
        candidates: List[Tuple[str, Path]] = []
        for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
            for label_path in hwmon.glob("in*_label"):
                label = _read_text(label_path).strip()
                number_match = re.search(r"in(\d+)_label$", label_path.name)
                if not number_match:
                    continue
                number = number_match.group(1)

                for power_name in (
                    f"in{number}_power_input",
                    f"power{number}_input",
                ):
                    power_path = hwmon / power_name
                    if power_path.exists():
                        candidates.append((label, power_path))
                        break

            for label_path in hwmon.glob("power*_label"):
                label = _read_text(label_path).strip()
                number_match = re.search(r"power(\d+)_label$", label_path.name)
                if not number_match:
                    continue
                power_path = hwmon / f"power{number_match.group(1)}_input"
                if power_path.exists():
                    candidates.append((label, power_path))

        return candidates

    @property
    def available(self) -> bool:
        return bool(self.gpu_load_paths or self.power_candidates)

    def _gpu_utilization(self) -> Optional[float]:
        for path in self.gpu_load_paths:
            raw = _read_number(path)
            if raw is None:
                continue
            # Common Jetson GPU load units are 0..1000.
            if raw > 100:
                raw = raw / 10.0
            return max(0.0, min(100.0, raw))
        return None

    @staticmethod
    def _normalize_power_w(raw: Optional[float]) -> Optional[float]:
        if raw is None:
            return None

        # hwmon power*_input is typically microwatts. Some Jetson INA nodes
        # expose milliwatts. Infer conservatively by magnitude.
        if raw > 100_000:
            return raw / 1_000_000.0
        if raw > 100:
            return raw / 1_000.0
        return raw

    def _power_by_labels(
        self,
        labels: set[str],
    ) -> Optional[float]:
        matches = [
            (label, path)
            for label, path in self.power_candidates
            if label.upper() in labels
        ]
        if not matches:
            return None
        return self._normalize_power_w(_read_number(matches[0][1]))

    def sample(self) -> Dict[str, Optional[float]]:
        return {
            "gpu_usage_percent": self._gpu_utilization(),
            "cpu_power_w": self._power_by_labels(self.CPU_POWER_LABELS),
            "gpu_power_w": self._power_by_labels(self.GPU_POWER_LABELS),
            "system_power_w": self._power_by_labels(self.TOTAL_POWER_LABELS),
        }


class ResourceMonitor:
    """
    Background client/server resource telemetry collector.

    The monitor never places telemetry on the measured application channel.
    It writes a local CSV sidecar plus a summary JSON file.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        monitor_cfg = config.get("resource_monitor", {})
        self.enabled = bool(monitor_cfg.get("enabled", True))
        self.interval_ms = int(monitor_cfg.get("interval_ms", 500))
        self.interval_sec = self.interval_ms / 1000.0
        self.network_interface = monitor_cfg.get("network_interface")
        self.gpu_index = int(monitor_cfg.get("gpu_index", 0))
        self.power_enabled = bool(monitor_cfg.get("power_enabled", True))

        if self.interval_ms < 100:
            raise ValueError("resource_monitor.interval_ms must be at least 100")

        self.experiment_id = config["experiment"]["experiment_id"]
        self.role = config["node"]["role"]
        self.device = config["device"]["label"]
        output_dir = Path(config["experiment"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        role_token = output_role_token(config)
        self.csv_path = (
            output_dir
            / f"{self.experiment_id}_{role_token}_resource.csv"
        )
        self.summary_path = (
            output_dir
            / f"{self.experiment_id}_{role_token}_resource_summary.json"
        )

        self.process = psutil.Process(os.getpid())
        self.cpu_count = max(psutil.cpu_count(logical=True) or 1, 1)
        self.network = NetworkCounter(self.network_interface)
        self.rapl = RaplCollector() if self.power_enabled else None
        self.nvidia = NvidiaSmiCollector(self.gpu_index)
        self.jetson = JetsonCollector()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_monotonic: Optional[float] = None
        self._sample_index = 0
        self._records: List[Dict[str, Any]] = []
        self._cpu_energy_integrated_j = 0.0
        self._gpu_energy_j = 0.0
        self._system_energy_j = 0.0
        self._last_energy_time: Optional[float] = None
        self._last_cpu_power_w: Optional[float] = None
        self._last_gpu_power_w: Optional[float] = None
        self._last_system_power_w: Optional[float] = None

        # Prime nonblocking psutil CPU counters.
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    @property
    def telemetry_sources(self) -> str:
        sources = ["psutil"]
        if self.rapl and self.rapl.available:
            sources.append("rapl")
        if self.nvidia.available:
            sources.append("nvidia-smi")
        if self.jetson.available:
            sources.append("jetson-sysfs")
        return "+".join(sources)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        self._stop_event.clear()
        self._start_monotonic = time.monotonic()
        self._last_energy_time = self._start_monotonic
        self._thread = threading.Thread(
            target=self._run,
            name=f"resource-monitor-{self.role}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(5.0, self.interval_sec * 4))
            self._thread = None

        # Capture one final sample after the workload exits.
        self._capture_sample()
        summary = self._build_summary()
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    def _run(self) -> None:
        # First sample is immediate so the baseline is visible in the CSV.
        while not self._stop_event.is_set():
            started = time.monotonic()
            self._capture_sample()
            elapsed = time.monotonic() - started
            wait = max(0.0, self.interval_sec - elapsed)
            if self._stop_event.wait(wait):
                break

    def _sample_gpu(self) -> Dict[str, Optional[float]]:
        nvidia = self.nvidia.sample()
        jetson = self.jetson.sample()

        return {
            "gpu_usage_percent": (
                nvidia.get("gpu_usage_percent")
                if nvidia.get("gpu_usage_percent") is not None
                else jetson.get("gpu_usage_percent")
            ),
            "gpu_memory_used_mb": nvidia.get("gpu_memory_used_mb"),
            "gpu_memory_total_mb": nvidia.get("gpu_memory_total_mb"),
            "cpu_power_w": jetson.get("cpu_power_w"),
            "gpu_power_w": (
                nvidia.get("gpu_power_w")
                if nvidia.get("gpu_power_w") is not None
                else jetson.get("gpu_power_w")
            ),
            "system_power_w": jetson.get("system_power_w"),
        }

    def _integrate_energy(
        self,
        now: float,
        cpu_power_w: Optional[float],
        gpu_power_w: Optional[float],
        system_power_w: Optional[float],
    ) -> None:
        if self._last_energy_time is None:
            self._last_energy_time = now
            self._last_cpu_power_w = cpu_power_w
            self._last_gpu_power_w = gpu_power_w
            self._last_system_power_w = system_power_w
            return

        delta_t = max(0.0, now - self._last_energy_time)

        if (
            cpu_power_w is not None
            and self._last_cpu_power_w is not None
        ):
            self._cpu_energy_integrated_j += (
                (cpu_power_w + self._last_cpu_power_w) / 2.0
            ) * delta_t

        if (
            gpu_power_w is not None
            and self._last_gpu_power_w is not None
        ):
            self._gpu_energy_j += (
                (gpu_power_w + self._last_gpu_power_w) / 2.0
            ) * delta_t

        if (
            system_power_w is not None
            and self._last_system_power_w is not None
        ):
            self._system_energy_j += (
                (system_power_w + self._last_system_power_w) / 2.0
            ) * delta_t

        self._last_energy_time = now
        self._last_cpu_power_w = cpu_power_w
        self._last_gpu_power_w = gpu_power_w
        self._last_system_power_w = system_power_w

    def _capture_sample(self) -> None:
        if not self.enabled:
            return

        now_mono = time.monotonic()
        now_ns = time.monotonic_ns()
        relative = (
            now_mono - self._start_monotonic
            if self._start_monotonic is not None
            else 0.0
        )

        network = self.network.sample()

        process_cpu_raw = self.process.cpu_percent(interval=None)
        process_cpu_normalized = process_cpu_raw / self.cpu_count
        system_cpu = psutil.cpu_percent(interval=None)

        memory_info = self.process.memory_info()
        process_memory_mb = memory_info.rss / (1024.0 * 1024.0)
        process_memory_percent = self.process.memory_percent()
        system_memory_percent = psutil.virtual_memory().percent

        gpu = self._sample_gpu()

        cpu_power_w = None
        cpu_energy_j = None
        if self.rapl is not None:
            cpu_power_w, cpu_energy_j = self.rapl.sample()

        if cpu_power_w is None:
            cpu_power_w = _safe_float(gpu.get("cpu_power_w"))

        gpu_power_w = _safe_float(gpu.get("gpu_power_w"))
        system_power_w = _safe_float(gpu.get("system_power_w"))

        self._integrate_energy(
            now=now_mono,
            cpu_power_w=cpu_power_w,
            gpu_power_w=gpu_power_w,
            system_power_w=system_power_w,
        )

        if cpu_energy_j is None and cpu_power_w is not None:
            cpu_energy_j = self._cpu_energy_integrated_j

        record: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "timestamp_utc": utc_now_iso(),
            "timestamp_monotonic_ns": now_ns,
            "relative_time_sec": relative,
            "sample_index": self._sample_index,
            "role": self.role,
            "device": self.device,
            "sample_interval_ms": self.interval_ms,
            "telemetry_source": self.telemetry_sources,
            **network,
            "cpu_usage_percent": process_cpu_normalized,
            "process_cpu_usage_percent_raw": process_cpu_raw,
            "system_cpu_usage_percent": system_cpu,
            "memory_usage_mb": process_memory_mb,
            "memory_usage_percent": process_memory_percent,
            "system_memory_usage_percent": system_memory_percent,
            "gpu_usage_percent": gpu.get("gpu_usage_percent"),
            "gpu_memory_used_mb": gpu.get("gpu_memory_used_mb"),
            "gpu_memory_total_mb": gpu.get("gpu_memory_total_mb"),
            "cpu_power_w": cpu_power_w,
            "gpu_power_w": gpu_power_w,
            "system_power_w": system_power_w,
            "cpu_energy_j": cpu_energy_j,
            "gpu_energy_j": (
                self._gpu_energy_j if gpu_power_w is not None else None
            ),
            "system_energy_j": (
                self._system_energy_j
                if system_power_w is not None
                else None
            ),
        }

        self._append_csv(record)
        self._records.append(record)
        self._sample_index += 1

    def _append_csv(self, record: Dict[str, Any]) -> None:
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=RESOURCE_FIELDS,
                extrasaction="ignore",
            )
            if new_file:
                writer.writeheader()
            writer.writerow(record)

    @staticmethod
    def _numeric_values(
        records: Iterable[Dict[str, Any]],
        field: str,
    ) -> List[float]:
        values: List[float] = []
        for record in records:
            value = _safe_float(record.get(field))
            if value is not None:
                values.append(value)
        return values

    def _metric_summary(self, field: str) -> Dict[str, Optional[float]]:
        values = self._numeric_values(self._records, field)
        if not values:
            return {"mean": None, "min": None, "max": None}
        return {
            "mean": mean(values),
            "min": min(values),
            "max": max(values),
        }

    def _build_summary(self) -> Dict[str, Any]:
        last = self._records[-1] if self._records else {}
        return {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "role": self.role,
            "device": self.device,
            "generated_at_utc": utc_now_iso(),
            "sample_interval_ms": self.interval_ms,
            "sample_count": len(self._records),
            "telemetry_source": self.telemetry_sources,
            "network_interface": last.get("network_interface"),
            "bytes_sent": last.get("bytes_sent", 0),
            "bytes_received": last.get("bytes_received", 0),
            "cpu_usage_percent": self._metric_summary(
                "cpu_usage_percent"
            ),
            "system_cpu_usage_percent": self._metric_summary(
                "system_cpu_usage_percent"
            ),
            "memory_usage_mb": self._metric_summary(
                "memory_usage_mb"
            ),
            "memory_usage_percent": self._metric_summary(
                "memory_usage_percent"
            ),
            "gpu_usage_percent": self._metric_summary(
                "gpu_usage_percent"
            ),
            "gpu_memory_used_mb": self._metric_summary(
                "gpu_memory_used_mb"
            ),
            "cpu_power_w": self._metric_summary("cpu_power_w"),
            "gpu_power_w": self._metric_summary("gpu_power_w"),
            "system_power_w": self._metric_summary("system_power_w"),
            "cpu_energy_j": last.get("cpu_energy_j"),
            "gpu_energy_j": last.get("gpu_energy_j"),
            "system_energy_j": last.get("system_energy_j"),
            "resource_csv": str(self.csv_path),
        }
