from __future__ import annotations

import csv
import datetime as dt
import json
import selectors
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .capture import (
    CaptureError,
    inspect_capture_interface,
    start_capture_process,
    stop_capture_process,
)
from .experiment_output import enforce_experiment_output_policy
from .offload import CaptureOffloadError, CaptureOffloadManager
from .traffic import extract_capture_artifacts


class ProxyError(RuntimeError):
    pass


DEFAULT_PROXY_CONFIG: Dict[str, Any] = {
    "experiment": {
        "experiment_id": "auto",
        "output_dir": "proxy_results",
        "existing_output_policy": "error",
    },
    "proxy": {
        "listen_host": "0.0.0.0",
        "listen_port": 5000,
        "upstream_host": "127.0.0.1",
        "upstream_port": 5001,
        "connect_timeout_sec": 30,
        "buffer_size": 65536,
        # recv() chunk boundaries are not packets. Keep only aggregate
        # forwarding counters by default to avoid large diagnostic CSVs.
        "forwarding_log_enabled": False,
    },
    "capture": {
        "enabled": True,
        "interface": "",
        # One or more client IPs are used only for network capture isolation.
        # They are stripped from classifier-ready packet sequences.
        "client_ip": None,  # legacy single-client field
        "client_ips": [],
        # Optional mapping such as {"10.42.0.47": "client_1"}.
        # Aliases are grouping metadata, never predictor features.
        "client_aliases": {},
        "strict_client_isolation": True,
        "per_client_artifacts": True,
        # Store only the first bytes of each frame to keep PCAPs manageable.
        # frame.len still reports the original on-wire frame length.
        "snaplen_bytes": 256,
        # Packet-size fidelity: disable GRO/GSO/TSO/LRO on the capture
        # interface for the duration of the experiment, verify, then restore.
        "offload_management": {
            "enabled": True,
            "required": True,
            "allow_sudo_noninteractive": True,
            "restore_on_exit": True,
            "features": ["gro", "gso", "tso", "lro"],
        },
        "proxy_client_facing_ip": None,
        "extract_after": True,
        "burst_gap_sec": 0.05,
        "idle_threshold_sec": 0.5,
        # Keep an overall row and additionally produce 5-second windows.
        "window_seconds": 5.0,
    },
}


def _deep_merge(
    base: Dict[str, Any],
    update: Dict[str, Any],
) -> Dict[str, Any]:
    result = json.loads(json.dumps(base))
    for key, value in update.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _new_experiment_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"EXP_PROXY_{stamp}"


def validate_proxy_config(config: Dict[str, Any]) -> None:
    forbidden = {
        "ai",
        "family",
        "architecture",
        "variant",
        "framework",
        "runtime",
        "application",
        "dataset",
        "device",
        "task",
        "deployment",
    }
    def collect_forbidden(
        value: Any,
    ) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key) in forbidden:
                    found.add(str(key))
                found.update(collect_forbidden(nested))
        elif isinstance(value, list):
            for nested in value:
                found.update(collect_forbidden(nested))
        return found

    present = collect_forbidden(config)
    if present:
        raise ProxyError(
            "Proxy configuration must remain label blind. "
            f"Remove ground-truth fields: {sorted(present)}"
        )

    policy = str(
        config.get("experiment", {}).get(
            "existing_output_policy",
            "error",
        )
    ).strip().lower()
    if policy not in {"error", "archive"}:
        raise ProxyError(
            "experiment.existing_output_policy must be "
            "error or archive"
        )

    proxy = config.get("proxy", {})
    capture = config.get("capture", {})

    listen_port = int(proxy.get("listen_port", 0))
    upstream_port = int(proxy.get("upstream_port", 0))
    if not (1 <= listen_port <= 65535):
        raise ProxyError("proxy.listen_port must be between 1 and 65535")
    if not (1 <= upstream_port <= 65535):
        raise ProxyError("proxy.upstream_port must be between 1 and 65535")

    if not str(proxy.get("upstream_host", "")).strip():
        raise ProxyError("proxy.upstream_host is required")

    buffer_size = int(proxy.get("buffer_size", 0))
    if buffer_size < 1024:
        raise ProxyError("proxy.buffer_size must be at least 1024 bytes")

    if bool(capture.get("enabled", True)):
        if not str(capture.get("interface", "")).strip():
            raise ProxyError(
                "capture.interface is required when capture.enabled=true"
            )

        client_ips = []
        legacy_client_ip = capture.get("client_ip")
        if legacy_client_ip:
            client_ips.append(str(legacy_client_ip).strip())
        configured_client_ips = capture.get("client_ips", []) or []
        if isinstance(configured_client_ips, str):
            configured_client_ips = [
                value.strip()
                for value in configured_client_ips.split(",")
                if value.strip()
            ]
        client_ips.extend(
            str(value).strip()
            for value in configured_client_ips
            if str(value).strip()
        )
        client_ips = list(dict.fromkeys(client_ips))

        if (
            bool(capture.get("strict_client_isolation", True))
            and not client_ips
        ):
            raise ProxyError(
                "capture.client_ips is required when capture is enabled. "
                "Specify every participating client IP so the BPF filter "
                "captures only client-facing traffic and excludes the "
                "proxy-to-upstream duplicate leg."
            )

        aliases = capture.get("client_aliases", {}) or {}
        if not isinstance(aliases, dict):
            raise ProxyError(
                "capture.client_aliases must be a mapping of client IP "
                "to capture alias/client ID"
            )
        unknown_alias_ips = sorted(
            str(ip)
            for ip in aliases
            if str(ip) not in client_ips
        )
        if unknown_alias_ips:
            raise ProxyError(
                "capture.client_aliases contains IPs not listed in "
                f"capture.client_ips: {unknown_alias_ips}"
            )

        snaplen_bytes = capture.get("snaplen_bytes", 256)
        if snaplen_bytes is not None and int(snaplen_bytes) < 0:
            raise ProxyError(
                "capture.snaplen_bytes must be nonnegative or null"
            )

        window_seconds = capture.get("window_seconds")
        if window_seconds is not None and float(window_seconds) <= 0:
            raise ProxyError(
                "capture.window_seconds must be positive or null"
            )

        offload_cfg = capture.get("offload_management", {}) or {}
        if not isinstance(offload_cfg, dict):
            raise ProxyError(
                "capture.offload_management must be a mapping"
            )
        features = offload_cfg.get(
            "features",
            ["gro", "gso", "tso", "lro"],
        )
        if isinstance(features, str):
            features = [
                value.strip().lower()
                for value in features.split(",")
                if value.strip()
            ]
        allowed = {"gro", "gso", "tso", "lro"}
        unknown = sorted(
            value for value in features if value not in allowed
        )
        if unknown:
            raise ProxyError(
                "capture.offload_management.features contains "
                f"unsupported values: {unknown}"
            )

        if (
            bool(offload_cfg.get("enabled", True))
            and not bool(
                offload_cfg.get("restore_on_exit", True)
            )
        ):
            raise ProxyError(
                "capture.offload_management.restore_on_exit must be true "
                "when automatic offload management is enabled"
            )


def load_proxy_config(
    path: str | Path,
) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ProxyError(f"Proxy configuration not found: {path}")

    supplied = yaml.safe_load(
        path.read_text(encoding="utf-8")
    ) or {}
    config = _deep_merge(DEFAULT_PROXY_CONFIG, supplied)

    if config["experiment"]["experiment_id"] in {
        None,
        "",
        "auto",
    }:
        config["experiment"]["experiment_id"] = (
            _new_experiment_id()
        )

    validate_proxy_config(config)
    return config


def save_proxy_config(
    config: Dict[str, Any],
    path: str | Path,
) -> None:
    validate_proxy_config(config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


class BlindTCPProxy:
    """
    Label-blind TCP byte-stream forwarder.

    The proxy never terminates TLS and never parses the application protocol.
    Encrypted bytes received from the client are forwarded unchanged to the
    upstream server, and server bytes are returned unchanged to the client.

    Packet-level research data comes from dumpcap/tshark on the client-facing
    interface. The forwarding CSV is diagnostic only: recv() chunk boundaries
    are not network packet boundaries and must not replace the PCAP-derived
    packet sequence for fingerprinting.
    """

    FORWARD_FIELDS = [
        "experiment_id",
        "timestamp_utc",
        "relative_time_sec",
        "connection_id",
        "direction",
        "chunk_bytes",
        "cumulative_up_bytes",
        "cumulative_down_bytes",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        validate_proxy_config(config)
        self.config = config
        self.stop_event = threading.Event()

        experiment = config["experiment"]
        self.experiment_id = str(
            experiment["experiment_id"]
        )
        self.output_dir = Path(
            experiment["output_dir"]
        )
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.forward_csv = (
            self.output_dir
            / f"{self.experiment_id}_proxy_forwarding.csv"
        )
        self.summary_json = (
            self.output_dir
            / f"{self.experiment_id}_proxy_summary.json"
        )
        self.pcap_path = (
            self.output_dir
            / f"{self.experiment_id}.pcapng"
        )

        self._start_monotonic = 0.0
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._capture_process = None
        self._capture_preflight: Dict[str, Any] = {}
        self._offload_manager: Optional[
            CaptureOffloadManager
        ] = None
        self._offload_state_path = (
            self.output_dir
            / f"{self.experiment_id}_proxy_offload_state.json"
        )

        self._connections = 0
        self._up_bytes = 0
        self._down_bytes = 0
        self._forward_rows = 0

    def _configured_client_ips(self) -> list[str]:
        capture = self.config.get("capture", {})
        values: list[str] = []
        legacy = capture.get("client_ip")
        if legacy:
            values.append(str(legacy).strip())
        configured = capture.get("client_ips", []) or []
        if isinstance(configured, str):
            configured = configured.split(",")
        values.extend(
            str(value).strip()
            for value in configured
            if str(value).strip()
        )
        return list(dict.fromkeys(values))

    def stop(self) -> None:
        self.stop_event.set()

    def serve_forever(self) -> Dict[str, Any]:
        enforce_experiment_output_policy(
            self.config,
            role="proxy",
        )
        self._start_monotonic = time.monotonic()

        if bool(
            self.config.get("proxy", {}).get(
                "forwarding_log_enabled", False
            )
        ):
            self._initialize_forward_csv()

        capture = self.config["capture"]
        listener: Optional[socket.socket] = None
        capture_result: Dict[str, Any] = {
            "capture_enabled": False,
        }

        try:
            if bool(capture.get("enabled", True)):
                interface = str(capture["interface"])
                client_ips = self._configured_client_ips()
                offload_cfg = (
                    capture.get("offload_management", {}) or {}
                )
                features = offload_cfg.get(
                    "features",
                    ["gro", "gso", "tso", "lro"],
                )
                if isinstance(features, str):
                    features = [
                        value.strip()
                        for value in features.split(",")
                        if value.strip()
                    ]

                self._offload_manager = CaptureOffloadManager(
                    interface=interface,
                    enabled=bool(
                        offload_cfg.get("enabled", True)
                    ),
                    required=bool(
                        offload_cfg.get("required", True)
                    ),
                    allow_sudo_noninteractive=bool(
                        offload_cfg.get(
                            "allow_sudo_noninteractive",
                            True,
                        )
                    ),
                    restore_on_exit=bool(
                        offload_cfg.get(
                            "restore_on_exit",
                            True,
                        )
                    ),
                    features=features,
                    state_path=self._offload_state_path,
                )

                offload_report = self._offload_manager.start()
                self._capture_preflight = inspect_capture_interface(
                    interface
                )
                self._capture_preflight[
                    "snaplen_bytes"
                ] = capture.get("snaplen_bytes", 256)
                self._capture_preflight[
                    "offload_management"
                ] = offload_report

                if self._offload_manager.status == "capture_safe":
                    print(
                        "[proxy] packet-size fidelity protection ACTIVE: "
                        "GRO/GSO/TSO/LRO verified disabled where supported "
                        f"on {interface}"
                    )
                elif not self._offload_manager.enabled:
                    print(
                        "[proxy] WARNING: capture offload management "
                        "is disabled"
                    )
                elif not self._offload_manager.required:
                    print(
                        "[proxy] WARNING: offload enforcement is "
                        "warning-only"
                    )

                for warning in self._capture_preflight.get(
                    "warnings", []
                ):
                    print(f"[proxy] WARNING: {warning}")

                self._capture_process = start_capture_process(
                    interface=interface,
                    output=str(self.pcap_path),
                    hosts=client_ips,
                    port=int(
                        self.config["proxy"]["listen_port"]
                    ),
                    snaplen_bytes=capture.get(
                        "snaplen_bytes", 256
                    ),
                )

            listener = self._create_listener()
            address = listener.getsockname()

            print(
                f"[proxy] listening on {address[0]}:{address[1]} "
                f"-> {self.config['proxy']['upstream_host']}:"
                f"{self.config['proxy']['upstream_port']}"
            )
            if self._capture_process is not None:
                print(
                    f"[proxy] capture interface="
                    f"{capture['interface']} pcap={self.pcap_path}"
                )
            print(
                "[proxy] TLS is forwarded end-to-end without decryption."
            )
            print("[proxy] press Ctrl+C to stop")

            while not self.stop_event.is_set():
                try:
                    client_sock, client_addr = listener.accept()
                except socket.timeout:
                    continue

                connection_id = uuid.uuid4().hex
                with self._lock:
                    self._connections += 1

                thread = threading.Thread(
                    target=self._handle_connection,
                    args=(
                        client_sock,
                        client_addr,
                        connection_id,
                    ),
                    daemon=True,
                )
                self._threads.append(thread)
                thread.start()

        except KeyboardInterrupt:
            print("\n[proxy] stopping...")
            self.stop_event.set()
        except CaptureOffloadError as exc:
            print(
                f"[proxy] capture offload protection failed: {exc}"
            )
            raise ProxyError(str(exc)) from exc
        finally:
            if listener is not None:
                try:
                    listener.close()
                except OSError:
                    pass

            for thread in list(self._threads):
                thread.join(timeout=2.0)

            capture_result = self._stop_and_extract_capture()
            self._restore_capture_offloads()

        summary = self._write_summary(capture_result)
        return summary

    def _restore_capture_offloads(self) -> None:
        manager = self._offload_manager
        if manager is None:
            return

        report = manager.restore()
        self._capture_preflight[
            "offload_management"
        ] = report

        if manager.changed_features:
            if report.get("status") == "restore_failed":
                print(
                    "[proxy] ERROR: failed to fully restore the original "
                    "capture-interface offload state. Inspect "
                    f"{self._offload_state_path}"
                )
            else:
                print(
                    "[proxy] restored original capture-interface "
                    "offload settings"
                )

    def _create_listener(self) -> socket.socket:
        proxy = self.config["proxy"]
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        sock.bind(
            (
                str(proxy["listen_host"]),
                int(proxy["listen_port"]),
            )
        )
        sock.listen(64)
        sock.settimeout(1.0)
        return sock

    def _handle_connection(
        self,
        client_sock: socket.socket,
        client_addr,
        connection_id: str,
    ) -> None:
        proxy = self.config["proxy"]
        upstream_sock: Optional[socket.socket] = None

        try:
            upstream_sock = socket.create_connection(
                (
                    str(proxy["upstream_host"]),
                    int(proxy["upstream_port"]),
                ),
                timeout=float(
                    proxy.get("connect_timeout_sec", 30)
                ),
            )
            client_sock.settimeout(None)
            upstream_sock.settimeout(None)

            self._forward_duplex(
                client_sock=client_sock,
                upstream_sock=upstream_sock,
                connection_id=connection_id,
            )
        except Exception as exc:
            print(
                f"[proxy] connection {connection_id[:8]} "
                f"from {client_addr} ended: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            for sock in (client_sock, upstream_sock):
                if sock is None:
                    continue
                try:
                    sock.close()
                except OSError:
                    pass

    def _forward_duplex(
        self,
        client_sock: socket.socket,
        upstream_sock: socket.socket,
        connection_id: str,
    ) -> None:
        selector = selectors.DefaultSelector()
        selector.register(
            client_sock,
            selectors.EVENT_READ,
            data=("up", upstream_sock),
        )
        selector.register(
            upstream_sock,
            selectors.EVENT_READ,
            data=("down", client_sock),
        )

        buffer_size = int(
            self.config["proxy"]["buffer_size"]
        )

        try:
            while not self.stop_event.is_set():
                events = selector.select(timeout=1.0)
                if not events:
                    continue

                for key, _ in events:
                    direction, destination = key.data
                    source = key.fileobj

                    data = source.recv(buffer_size)
                    if not data:
                        return

                    destination.sendall(data)
                    self._record_forward(
                        connection_id=connection_id,
                        direction=direction,
                        chunk_bytes=len(data),
                    )
        finally:
            selector.close()

    def _initialize_forward_csv(self) -> None:
        with self.forward_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.FORWARD_FIELDS,
            )
            writer.writeheader()

    def _record_forward(
        self,
        connection_id: str,
        direction: str,
        chunk_bytes: int,
    ) -> None:
        now = time.monotonic()

        with self._lock:
            if direction == "up":
                self._up_bytes += chunk_bytes
            else:
                self._down_bytes += chunk_bytes
            self._forward_rows += 1

            row = {
                "experiment_id": self.experiment_id,
                "timestamp_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
                "relative_time_sec": (
                    now - self._start_monotonic
                ),
                "connection_id": connection_id,
                "direction": direction,
                "chunk_bytes": chunk_bytes,
                "cumulative_up_bytes": self._up_bytes,
                "cumulative_down_bytes": self._down_bytes,
            }

            if bool(
                self.config.get("proxy", {}).get(
                    "forwarding_log_enabled", False
                )
            ):
                with self.forward_csv.open(
                    "a",
                    newline="",
                    encoding="utf-8",
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=self.FORWARD_FIELDS,
                    )
                    writer.writerow(row)

    def _stop_and_extract_capture(
        self,
    ) -> Dict[str, Any]:
        capture = self.config["capture"]

        if self._capture_process is None:
            return {
                "capture_enabled": False,
            }

        stop_capture_process(self._capture_process)
        self._capture_process = None

        # Capture has ended. Restore host networking before CPU-heavy
        # post-processing/extraction so the modified NIC state exists only
        # during the measured interval.
        self._restore_capture_offloads()

        result: Dict[str, Any] = {
            "capture_enabled": True,
            "pcap": str(self.pcap_path),
        }

        if not bool(capture.get("extract_after", True)):
            return result

        if (
            not self.pcap_path.exists()
            or self.pcap_path.stat().st_size == 0
        ):
            result["capture_error"] = (
                "PCAP is missing or empty"
            )
            return result

        server_ip = (
            capture.get("proxy_client_facing_ip")
            or None
        )
        client_ip = capture.get("client_ip") or None

        try:
            artifacts = extract_capture_artifacts(
                pcap_path=self.pcap_path,
                experiment_id=self.experiment_id,
                output_dir=str(self.output_dir),
                server_ip=server_ip,
                client_ip=client_ip,
                client_ips=self._configured_client_ips(),
                client_aliases=(
                    capture.get("client_aliases", {}) or {}
                ),
                per_client_artifacts=bool(
                    capture.get("per_client_artifacts", True)
                ),
                capture_interface=str(capture.get("interface") or ""),
                capture_preflight=self._capture_preflight,
                burst_gap_sec=float(
                    capture.get("burst_gap_sec", 0.05)
                ),
                idle_threshold_sec=float(
                    capture.get(
                        "idle_threshold_sec",
                        0.5,
                    )
                ),
                window_seconds=capture.get(
                    "window_seconds"
                ),
            )
            result.update(artifacts)
        except Exception as exc:
            result["capture_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        return result

    def _write_summary(
        self,
        capture_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        elapsed = max(
            0.0,
            time.monotonic() - self._start_monotonic,
        )

        summary = {
            "schema_version": "1.0",
            "experiment_id": self.experiment_id,
            "role": "proxy",
            "label_blind": True,
            "tls_termination": False,
            "application_payload_parsing": False,
            "duration_sec": elapsed,
            "connections": self._connections,
            "forwarding_chunks_observed": self._forward_rows,
            "forwarding_log_enabled": bool(
                self.config.get("proxy", {}).get(
                    "forwarding_log_enabled", False
                )
            ),
            "bytes_up": self._up_bytes,
            "bytes_down": self._down_bytes,
            "forwarding_csv": (
                str(self.forward_csv)
                if bool(
                    self.config.get("proxy", {}).get(
                        "forwarding_log_enabled", False
                    )
                )
                else None
            ),
            "capture_offload_management": (
                self._offload_manager.report()
                if self._offload_manager is not None
                else None
            ),
            "capture_isolation": {
                "client_ips_configured": len(
                    self._configured_client_ips()
                ),
                "strict_client_isolation": bool(
                    self.config.get("capture", {}).get(
                        "strict_client_isolation", True
                    )
                ),
                "capture_filter_scope": "client_facing_only",
            },
            "capture": capture_result,
        }

        self.summary_json.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return summary
