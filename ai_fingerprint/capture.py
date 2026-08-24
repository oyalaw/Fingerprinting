from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .traffic import extract_capture_artifacts


class CaptureError(RuntimeError):
    pass


def build_capture_filter(
    host: Optional[str],
    port: Optional[int],
) -> str:
    terms = []
    if host:
        terms.append(f"host {host}")
    if port:
        terms.append(f"port {int(port)}")
    return " and ".join(terms)


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

    capture_filter = build_capture_filter(host, port)
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
    experiment_id: Optional[str] = None,
    extract_after: bool = True,
    output_dir: Optional[str] = None,
    server_ip: Optional[str] = None,
    client_ip: Optional[str] = None,
    burst_gap_sec: float = 0.05,
    idle_threshold_sec: float = 0.5,
    window_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    output_path = Path(output)
    process = start_capture_process(
        interface=interface,
        output=str(output_path),
        host=host,
        port=port,
    )
    print("[capture] press Ctrl+C to stop")
    interrupted = False
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

    result: Dict[str, Any] = {
        "pcap": str(output_path),
        "capture_return_code": return_code,
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

    direction_server_ip = server_ip or host

    print("[capture] extracting packet sequence and handcrafted features...")
    artifacts = extract_capture_artifacts(
        pcap_path=output_path,
        experiment_id=experiment_id,
        output_dir=output_dir or str(output_path.parent),
        server_ip=direction_server_ip,
        client_ip=client_ip,
        burst_gap_sec=burst_gap_sec,
        idle_threshold_sec=idle_threshold_sec,
        window_seconds=window_seconds,
    )
    result.update(artifacts)
    return result
