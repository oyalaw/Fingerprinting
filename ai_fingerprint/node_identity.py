"""
Per-machine identity: which client this machine is, and its device and operating-system labels.

The identity is set once per machine (`python3 main.py identity`) and stored outside the
repository, at ~/.ai_fingerprint/node_identity.yaml (or $AIFP_NODE_IDENTITY), so it survives
code updates. `configure` then fills client_id, device.label, and device.operating_system from
it instead of asking, and `run` refuses a config whose identity differs from this machine's,
so a typing error or a config copied to the wrong machine cannot mislabel a trace.

Device and operating system are detected from the system where possible (Jetson model from the
device tree, vendor and laptop/desktop from DMI and the battery, Linux distribution from
/etc/os-release, Windows 10/11 from the build number).
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CLIENT_IDS = [f"client_{i}" for i in range(1, 10)]
ROLES = ["client", "server", "proxy"]


def identity_path() -> Path:
    env = os.environ.get("AIFP_NODE_IDENTITY")
    return Path(env).expanduser() if env else Path.home() / ".ai_fingerprint" / "node_identity.yaml"


def load_identity() -> Optional[Dict[str, Any]]:
    path = identity_path()
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def save_identity(identity: Dict[str, Any]) -> Path:
    path = identity_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(identity, sort_keys=False), encoding="utf-8")
    return path


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
    except Exception:
        return ""


def _windows_query(command: str) -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                             capture_output=True, text=True, timeout=10, check=False)
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _has_battery() -> bool:
    try:
        import psutil
        return psutil.sensors_battery() is not None
    except Exception:
        return False


def detect_device_label() -> Optional[str]:
    model = _read("/proc/device-tree/model").lower()
    if "jetson" in model or "orin" in model or "xavier" in model:
        if "agx" in model and "orin" in model:
            return "jetson_agx_orin"
        if "orin" in model:
            return "jetson_orin_nano"
        if "xavier" in model:
            return "jetson_xavier"
    if "raspberry pi 5" in model:
        return "raspberry_pi_5"
    if "dgx" in _read("/sys/class/dmi/id/product_name").lower() and "spark" in _read(
            "/sys/class/dmi/id/product_name").lower():
        return "nvidia_dgx_spark"
    system = platform.system()
    if system == "Windows":
        vendor = _windows_query("(Get-CimInstance Win32_ComputerSystem).Manufacturer").lower()
    else:
        vendor = _read("/sys/class/dmi/id/sys_vendor").lower()
    laptop = _has_battery()
    if "dell" in vendor:
        return "dell_laptop" if laptop else "dell_desktop"
    if system == "Darwin":
        return "macbook"
    if system == "Windows":
        return "generic_windows_desktop"
    if system == "Linux":
        return "generic_linux_desktop"
    return None


def detect_operating_system() -> Optional[str]:
    system = platform.system()
    if system == "Windows":
        build = re.search(r"(\d+)$", platform.version() or "")
        return "windows_11" if build and int(build.group(1)) >= 22000 else "windows_10"
    if system == "Darwin":
        return "macos"
    if system == "Linux":
        ids = dict(re.findall(r'^(\w+)="?([^"\n]*)"?', _read("/etc/os-release"), re.M))
        known = {"ubuntu": "ubuntu", "debian": "debian", "fedora": "fedora", "rhel": "rhel",
                 "rocky": "rocky_linux", "almalinux": "almalinux", "arch": "arch_linux",
                 "raspbian": "raspberry_pi_os"}
        return known.get(ids.get("ID", "").lower(), "other_linux")
    return None


def detected_identity() -> Dict[str, Any]:
    return {"device_label": detect_device_label(), "operating_system": detect_operating_system()}


def identity_mismatches(config: Dict[str, Any], identity: Optional[Dict[str, Any]] = None) -> list[str]:
    """Differences between a config and this machine's saved identity (empty if none saved)."""
    identity = identity if identity is not None else load_identity()
    if not identity:
        return []
    problems = []
    device = config.get("device", {}) or {}
    for key, cfg_value in (("device_label", device.get("label")),
                           ("operating_system", device.get("operating_system"))):
        want = identity.get(key)
        if want and cfg_value and want != cfg_value:
            problems.append(f"{key} is {cfg_value} in the config but {want} on this machine ({identity_path()})")
    role = (config.get("node", {}) or {}).get("role")
    if role == "client" and identity.get("role") == "client":
        cid = (config.get("federated", {}) or {}).get("client_id")
        want = identity.get("client_id")
        if want and cid and want != cid:
            problems.append(f"client_id is {cid} in the config but this machine is {want} ({identity_path()})")
    return problems
