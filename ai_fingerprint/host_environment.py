"""
Host environment checks and logging for experiment roles.

* host_identity / clock_status / power_mode   recorded in reproducibility.json at start-up
* device_preflight                            refuses to start a role on the wrong machine
                                              (e.g. the laptop's config on a Jetson)
* WifiLogger                                  samples Wi-Fi link quality to <exp>_<role>_wifi.csv

Every probe runs an operating-system tool with a short timeout and never raises: a missing tool
yields None fields and the raw message, so logging can never stop an experiment.
"""
from __future__ import annotations

import csv
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WIFI_FIELDS = [
    "timestamp_utc", "monotonic_sec", "interface", "ssid", "signal_dbm", "signal_percent",
    "tx_rate_mbps", "rx_rate_mbps", "frequency_mhz", "channel", "source",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (res.stdout or "") + (res.stderr or "")
    except Exception as exc:  # pragma: no cover - defensive
        return f"{type(exc).__name__}: {exc}"


def _float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text or "", re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def host_identity() -> Dict[str, Any]:
    return {"hostname": socket.gethostname(), "system": platform.system(),
            "machine": platform.machine(), "release": platform.release()}


def clock_status() -> Dict[str, Any]:
    """Whether the system clock is synchronized, and the offset when the tool reports it.
    Cross-device window alignment relies on synchronized clocks on all roles."""
    system = platform.system()
    if system == "Windows":
        out = _run(["w32tm", "/query", "/status"])
        if out is None:
            return {"source": None, "synchronized": None, "offset_sec": None, "raw": None}
        source = re.search(r"Source:\s*(.+)", out)
        synced = None if "denied" in out.lower() else bool(source and "Local CMOS" not in source.group(1)
                                                          and "Free-running" not in source.group(1))
        return {"source": "w32tm", "synchronized": synced,
                "offset_sec": _float(r"Phase Offset:\s*([-\d.,]+)s", out), "raw": out.strip()[:800]}
    out = _run(["chronyc", "tracking"])
    if out and "Leap status" in out:
        return {"source": "chronyc", "synchronized": "Not synchronised" not in out,
                "offset_sec": _float(r"System time\s*:\s*([\d.]+) seconds", out), "raw": out.strip()[:800]}
    out = _run(["timedatectl", "show", "-p", "NTPSynchronized", "-p", "NTP"])
    if out and "NTPSynchronized" in out:
        return {"source": "timedatectl", "synchronized": "NTPSynchronized=yes" in out,
                "offset_sec": None, "raw": out.strip()}
    return {"source": None, "synchronized": None, "offset_sec": None, "raw": out}


def power_mode() -> Dict[str, Any]:
    """Jetson power mode (nvpmodel) and L4T release; empty on other hosts."""
    info: Dict[str, Any] = {}
    rel = Path("/etc/nv_tegra_release")
    if rel.exists():
        try:
            info["l4t_release"] = rel.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
        except Exception:
            pass
    out = _run(["nvpmodel", "-q"])
    if out:
        m = re.search(r"NV Power Mode:\s*(.+)", out)
        info["nvpmodel"] = m.group(1).strip() if m else out.strip()[:200]
    return info


def device_preflight(config: Dict[str, Any]) -> List[str]:
    """Problems that mean this config is being started on the wrong machine (empty list = OK)."""
    problems: List[str] = []
    device = config.get("device", {}) or {}
    label = str(device.get("label", "")).lower()
    os_cfg = str(device.get("operating_system", "")).lower()
    system, machine = platform.system(), platform.machine().lower()
    if os_cfg.startswith("windows") and system != "Windows":
        problems.append(f"config is for a Windows device ({label}) but this host runs {system}")
    if os_cfg in ("ubuntu", "linux", "other_linux") and system != "Linux":
        problems.append(f"config is for a Linux device ({label}) but this host runs {system}")
    is_jetson = Path("/etc/nv_tegra_release").exists() or "tegra" in platform.release().lower()
    if "jetson" in label and not is_jetson:
        problems.append(f"config is for a Jetson ({label}) but this host is not a Jetson ({machine})")
    if "jetson" not in label and label and is_jetson:
        problems.append(f"this host is a Jetson but the config is for {label}")
    campaign = config.get("campaign", {}) or {}
    planned = campaign.get("planned_device")
    if planned and planned != device.get("label"):
        problems.append(f"device.label {device.get('label')} differs from the planned device {planned}")
    host = campaign.get("expected_host")
    if host and host != socket.gethostname():
        problems.append(f"planned for host {host} but this host is {socket.gethostname()}")
    from .node_identity import identity_mismatches
    problems.extend(identity_mismatches(config))
    if campaign.get("require_clock_sync") and clock_status().get("synchronized") is False:
        problems.append("system clock is not synchronized (NTP/chrony/w32tm)")
    return problems


def wifi_sample(interface: Optional[str]) -> Dict[str, Any]:
    """One reading of Wi-Fi link quality for the given interface."""
    row: Dict[str, Any] = {k: None for k in WIFI_FIELDS}
    row.update({"timestamp_utc": _utc_now(), "monotonic_sec": time.monotonic(), "interface": interface})
    if platform.system() == "Windows":
        out = _run(["netsh", "wlan", "show", "interfaces"]) or ""
        blocks = re.split(r"\n\s*\n", out)
        block = next((b for b in blocks if interface and re.search(rf"Name\s*:\s*{re.escape(interface)}\s*$", b,
                                                                     re.MULTILINE)), out)
        ssid = re.search(r"^\s*SSID\s*:\s*(.+)$", block, re.MULTILINE)
        row.update({"ssid": ssid.group(1).strip() if ssid else None,
                    "signal_percent": _float(r"Signal\s*:\s*(\d+)%", block),
                    "rx_rate_mbps": _float(r"Receive rate \(Mbps\)\s*:\s*([\d.]+)", block),
                    "tx_rate_mbps": _float(r"Transmit rate \(Mbps\)\s*:\s*([\d.]+)", block),
                    "channel": _float(r"Channel\s*:\s*(\d+)", block), "source": "netsh" if out else None})
        return row
    if interface:
        out = _run(["iw", "dev", interface, "link"])
        if out and "Connected" in out:
            ssid = re.search(r"SSID:\s*(.+)", out)
            row.update({"ssid": ssid.group(1).strip() if ssid else None,
                        "signal_dbm": _float(r"signal:\s*(-?\d+)", out),
                        "tx_rate_mbps": _float(r"tx bitrate:\s*([\d.]+)", out),
                        "rx_rate_mbps": _float(r"rx bitrate:\s*([\d.]+)", out),
                        "frequency_mhz": _float(r"freq:\s*([\d.]+)", out), "source": "iw"})
            return row
    out = _run(["nmcli", "-t", "-f", "ACTIVE,SSID,SIGNAL,CHAN,RATE,DEVICE", "dev", "wifi"])
    for line in (out or "").splitlines():
        parts = line.split(":")
        if parts and parts[0] == "yes":
            row.update({"ssid": parts[1] if len(parts) > 1 else None,
                        "signal_percent": _float(r"(\d+)", parts[2]) if len(parts) > 2 else None,
                        "channel": _float(r"(\d+)", parts[3]) if len(parts) > 3 else None,
                        "tx_rate_mbps": _float(r"([\d.]+)", parts[4]) if len(parts) > 4 else None,
                        "source": "nmcli"})
            break
    return row


class WifiLogger:
    """Background thread appending wifi_sample() rows to a CSV every interval_sec."""

    def __init__(self, csv_path: Path, interface: Optional[str], interval_sec: float = 30.0) -> None:
        self.csv_path = Path(csv_path)
        self.interface = interface
        self.interval_sec = max(5.0, float(interval_sec))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="wifi-logger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=10.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                row = wifi_sample(self.interface)
                new = not self.csv_path.exists()
                with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=WIFI_FIELDS, extrasaction="ignore")
                    if new:
                        w.writeheader()
                    w.writerow(row)
            except Exception:
                pass
            self._stop.wait(self.interval_sec)
