from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .offload import CaptureOffloadManager, read_offload_state
from .traffic import extract_capture_artifacts


class CaptureError(RuntimeError):
    pass


def build_capture_filter(
    host: Optional[str] = None,
    port: Optional[int] = None,
    hosts: Optional[Sequence[str]] = None,
) -> str:
    """Build a BPF capture filter.

    ``hosts`` is used by the inline proxy to isolate one or more client
    endpoints. This prevents the proxy-to-upstream copy of the same byte
    stream from being captured a second time when both legs share the same
    interface and port. ``host`` is retained for backwards compatibility.
    """
    normalized_hosts = []
    if hosts:
        normalized_hosts.extend(
            str(value).strip()
            for value in hosts
            if str(value).strip()
        )
    if host and str(host).strip():
        normalized_hosts.append(str(host).strip())

    # Preserve order while removing duplicates.
    normalized_hosts = list(dict.fromkeys(normalized_hosts))

    terms = []
    if len(normalized_hosts) == 1:
        terms.append(f"host {normalized_hosts[0]}")
    elif len(normalized_hosts) > 1:
        host_term = " or ".join(
            f"host {value}"
            for value in normalized_hosts
        )
        terms.append(f"({host_term})")

    if port:
        terms.append(f"port {int(port)}")
    return " and ".join(terms)


def inspect_capture_interface(interface: str) -> Dict[str, Any]:
    """Read MTU and capture-affecting offload state without changing it."""
    result: Dict[str, Any] = {
        "interface": interface,
        "mtu": None,
        "offloads": {},
        "offload_details": {},
        "possible_capture_coalescing": False,
        "warnings": [],
    }

    mtu_path = Path("/sys/class/net") / interface / "mtu"
    try:
        result["mtu"] = int(mtu_path.read_text().strip())
    except Exception:
        pass

    state = read_offload_state(interface)
    result["offload_probe"] = state

    features = state.get("features", {}) or {}
    long_names = {
        "gro": "generic-receive-offload",
        "gso": "generic-segmentation-offload",
        "tso": "tcp-segmentation-offload",
        "lro": "large-receive-offload",
    }
    for short_name, item in features.items():
        long_name = long_names.get(short_name, short_name)
        enabled = item.get("enabled")
        if enabled is not None:
            result["offloads"][long_name] = bool(enabled)
        result["offload_details"][short_name] = item

    enabled_offloads = [
        name
        for name, enabled in result["offloads"].items()
        if enabled
    ]
    if enabled_offloads:
        result["possible_capture_coalescing"] = True
        result["warnings"].append(
            "Capture-affecting offloads are enabled: "
            + ", ".join(enabled_offloads)
            + ". GRO/GSO/TSO/LRO can distort packet-size observations."
        )

    if not state.get("read_ok"):
        result["warnings"].append(
            "Capture offload state could not be verified: "
            + str(state.get("error", "unknown error"))
        )

    return result


def _stop_capture_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    try:
        if os.name == "nt":
            process.terminate()
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=10)
    except Exception:
        process.kill()
        process.wait(timeout=5)


def start_capture_process(
    interface: str,
    output: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    hosts: Optional[Sequence[str]] = None,
    snaplen_bytes: Optional[int] = None,
) -> subprocess.Popen:
    dumpcap = shutil.which("dumpcap")
    tshark = shutil.which("tshark")
    executable = dumpcap or tshark
    if not executable:
        raise CaptureError(
            "Neither dumpcap nor tshark is available on this system"
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    command = [
        executable,
        "-i",
        interface,
        "-w",
        str(output_path),
    ]

    if snaplen_bytes is not None and int(snaplen_bytes) > 0:
        # The original on-wire frame length remains available in the PCAP
        # record even when only the first N bytes are stored. Fingerprinting
        # uses frame.len/timing/direction rather than encrypted payload bytes.
        command.extend(["-s", str(int(snaplen_bytes))])

    capture_filter = build_capture_filter(host=host, port=port, hosts=hosts)
    if capture_filter:
        command.extend(["-f", capture_filter])

    print("[capture] command:", " ".join(command))
    process = subprocess.Popen(command)

    # Detect immediate startup failures such as missing capture privileges,
    # invalid interfaces, or an unavailable device before the experiment
    # proceeds without a usable PCAP.
    time.sleep(0.25)
    return_code = process.poll()
    if return_code is not None and return_code != 0:
        raise CaptureError(
            "Capture process failed to start successfully. "
            f"Return code: {return_code}. Check interface name and "
            "dumpcap/tshark capture permissions."
        )

    return process


def stop_capture_process(process: subprocess.Popen) -> None:
    _stop_capture_process(process)


def run_capture(
    interface: str,
    output: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    hosts: Optional[Sequence[str]] = None,
    snaplen_bytes: Optional[int] = 256,
    experiment_id: Optional[str] = None,
    extract_after: bool = True,
    output_dir: Optional[str] = None,
    server_ip: Optional[str] = None,
    client_ip: Optional[str] = None,
    burst_gap_sec: float = 0.05,
    idle_threshold_sec: float = 0.5,
    window_seconds: Optional[float] = None,
    manage_offloads: bool = True,
    require_offloads_disabled: bool = True,
    allow_sudo_noninteractive: bool = True,
) -> Dict[str, Any]:
    output_path = Path(output)

    state_path = (
        output_path.parent
        / f"{experiment_id or output_path.stem}_capture_offload_state.json"
    )
    manager = CaptureOffloadManager(
        interface=interface,
        enabled=manage_offloads,
        required=require_offloads_disabled,
        allow_sudo_noninteractive=allow_sudo_noninteractive,
        restore_on_exit=True,
        state_path=state_path,
    )

    process = None
    return_code = 0
    interrupted = False

    try:
        offload_start = manager.start()
        capture_preflight = inspect_capture_interface(interface)
        capture_preflight["snaplen_bytes"] = snaplen_bytes
        capture_preflight["offload_management"] = offload_start

        if manager.status == "capture_safe":
            print(
                "[capture] packet-size fidelity protection ACTIVE "
                f"on {interface}: GRO/GSO/TSO/LRO verified disabled "
                "where supported"
            )
        elif not manage_offloads:
            print(
                "[capture] WARNING: automatic offload management is disabled"
            )
        elif not require_offloads_disabled:
            print(
                "[capture] WARNING: offload verification is warning-only"
            )

        for warning in capture_preflight.get("warnings", []):
            print(f"[capture] WARNING: {warning}")

        process = start_capture_process(
            interface=interface,
            output=str(output_path),
            host=host,
            port=port,
            hosts=hosts,
            snaplen_bytes=snaplen_bytes,
        )
        print("[capture] press Ctrl+C to stop")

        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            print("\n[capture] stopping capture gracefully...")
            _stop_capture_process(process)
            return_code = process.returncode or 0

        if not interrupted and return_code != 0:
            raise CaptureError(
                f"Capture process exited with return code {return_code}"
            )

    finally:
        if process is not None and process.poll() is None:
            _stop_capture_process(process)

        restore_report = manager.restore()
        if manager.changed_features:
            if restore_report.get("status") == "restore_failed":
                print(
                    "[capture] ERROR: one or more interface offload "
                    "settings could not be restored. Inspect "
                    f"{state_path}"
                )
            else:
                print(
                    "[capture] restored original capture-interface "
                    "offload settings"
                )

    result: Dict[str, Any] = {
        "pcap": str(output_path),
        "capture_return_code": return_code,
        "offload_state_json": str(state_path),
        "offload_management": manager.report(),
    }

    if not extract_after:
        return result

    if not experiment_id:
        raise CaptureError(
            "experiment_id is required when automatic extraction is enabled"
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise CaptureError(
            f"Capture file is missing or empty: {output_path}"
        )

    # Preserve the interface snapshot taken while capture protection was
    # active; only update the lifecycle report with the verified restoration.
    capture_preflight["offload_management"] = manager.report()

    direction_server_ip = server_ip or host

    print("[capture] extracting packet sequence and handcrafted features...")
    artifacts = extract_capture_artifacts(
        pcap_path=output_path,
        experiment_id=experiment_id,
        output_dir=output_dir or str(output_path.parent),
        server_ip=direction_server_ip,
        client_ip=client_ip,
        client_ips=list(hosts or []),
        capture_interface=interface,
        capture_preflight=capture_preflight,
        burst_gap_sec=burst_gap_sec,
        idle_threshold_sec=idle_threshold_sec,
        window_seconds=window_seconds,
    )
    result.update(artifacts)
    return result
