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
    build_capture_filter,
    inspect_capture_interface,
    start_capture_process,
    stop_capture_process,
)
from .live_inference import (
    LiveArchitectureMonitor,
    run_final_architecture_inference,
)
from .experiment_output import enforce_experiment_output_policy
from .offload import CaptureOffloadError, CaptureOffloadManager
from .traffic import extract_capture_artifacts


class ProxyError(RuntimeError):
    pass


DEFAULT_PROXY_CONFIG: Dict[str, Any] = {
    "experiment": {
        "experiment_id": "auto",
        "output_dir": "experiments/results",
        "results_root": "experiments/results",
        "storage_locator": None,
        "existing_output_policy": "error",
    },
    "proxy": {
        "listen_host": "10.42.0.1",
        "listen_port": 8080,
        "upstream_host": "10.42.0.195",
        "upstream_port": 8080,
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
        # Automatic mode discovers participating clients from accepted proxy
        # connections while the BPF excludes the known upstream FL server.
        # Manual mode retains the legacy allow-list behavior.
        "client_discovery_mode": "automatic",
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
        # Legacy single-scale setting retained for compatibility.
        "window_seconds": 5.0,
        # Shared scales for real-time and end-of-run feature extraction.
        "window_sizes_sec": [0.5, 1.0, 2.0, 5.0],
        # Prevent an old persisted 5-second-only config from silently
        # defeating the multiscale real-time experiment. Set true only when
        # a deliberate single-scale study is intended.
        "allow_single_scale": False,
    },
    "architecture_inference": {
        "enabled": True,
        "realtime_enabled": True,
        "final_enabled": True,
        "realtime_required": True,
        "model_root": "fingerprinting_models",
        "feature_modes": ["full", "size_normalized"],
        "confidence_threshold": 0.90,
        "stable_windows": 3,
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

        discovery_mode = str(
            capture.get("client_discovery_mode", "automatic")
        ).strip().lower()
        if discovery_mode not in {"automatic", "manual"}:
            raise ProxyError(
                "capture.client_discovery_mode must be automatic or manual"
            )
        if (
            discovery_mode == "manual"
            and bool(capture.get("strict_client_isolation", True))
            and not client_ips
        ):
            raise ProxyError(
                "capture.client_ips is required when "
                "capture.client_discovery_mode=manual. In automatic mode "
                "the proxy discovers client IPs from accepted connections "
                "and excludes the configured upstream server from capture."
            )

        if discovery_mode == "automatic":
            upstream_host = str(proxy.get("upstream_host", "")).strip()
            if not upstream_host:
                raise ProxyError(
                    "proxy.upstream_host is required for automatic client "
                    "discovery so the upstream duplicate leg can be excluded"
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
        if discovery_mode == "manual" and unknown_alias_ips:
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

        window_sizes = capture.get("window_sizes_sec", []) or []
        if isinstance(window_sizes, (int, float, str)):
            window_sizes = [window_sizes]
        for value in window_sizes:
            if float(value) <= 0:
                raise ProxyError(
                    "capture.window_sizes_sec values must be positive"
                )

        inference = config.get("architecture_inference", {}) or {}
        if (
            bool(inference.get("enabled", False))
            and bool(inference.get("realtime_enabled", True))
            and len(window_sizes) < 2
            and not bool(capture.get("allow_single_scale", False))
        ):
            raise ProxyError(
                "Real-time architecture fingerprinting requires multiscale "
                "capture windows by default. Use capture.window_sizes_sec="
                "[0.5, 1.0, 2.0, 5.0], or set "
                "capture.allow_single_scale=true for an intentional "
                "single-scale experiment."
            )

        if bool(inference.get("enabled", False)):
            modes = inference.get(
                "feature_modes",
                ["full", "size_normalized"],
            )
            if not isinstance(modes, list) or not modes:
                raise ProxyError(
                    "architecture_inference.feature_modes must be a "
                    "non-empty list"
                )
            unknown_modes = sorted(
                set(modes) - {"full", "size_normalized"}
            )
            if unknown_modes:
                raise ProxyError(
                    "Unknown architecture feature modes: "
                    f"{unknown_modes}"
                )
            threshold = float(
                inference.get("confidence_threshold", 0.90)
            )
            if not 0.0 < threshold <= 1.0:
                raise ProxyError(
                    "architecture_inference.confidence_threshold must "
                    "be in (0, 1]"
                )
            if int(inference.get("stable_windows", 3)) < 1:
                raise ProxyError(
                    "architecture_inference.stable_windows must be >= 1"
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
        self._live_architecture_monitor: Optional[
            LiveArchitectureMonitor
        ] = None
        self._live_architecture_summary: Dict[str, Any] = {}
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

        self._discovered_client_ips: list[str] = []
        self._discovered_client_aliases: Dict[str, str] = {}
        for ip in self._configured_client_ips():
            self._register_discovered_client(
                ip,
                source="configured",
                notify_monitor=False,
            )

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

    def _client_discovery_mode(self) -> str:
        return str(
            self.config.get("capture", {}).get(
                "client_discovery_mode", "automatic"
            )
        ).strip().lower()

    def _capture_client_ips(self) -> list[str]:
        with self._lock:
            values = list(self._discovered_client_ips)
        if not values:
            values = self._configured_client_ips()
        return values

    def _capture_client_aliases(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._discovered_client_aliases)

    def _register_discovered_client(
        self,
        client_ip: str,
        *,
        source: str,
        notify_monitor: bool = True,
    ) -> str:
        value = str(client_ip).strip()
        if not value:
            raise ProxyError("Discovered client IP is empty")

        upstream_host = str(
            self.config.get("proxy", {}).get("upstream_host", "")
        ).strip()
        listen_host = str(
            self.config.get("proxy", {}).get("listen_host", "")
        ).strip()
        if value in {upstream_host, listen_host}:
            return ""

        created = False
        with self._lock:
            existing = self._discovered_client_aliases.get(value)
            if existing:
                alias = existing
            else:
                configured_alias = str(
                    (self.config.get("capture", {}).get(
                        "client_aliases", {}
                    ) or {}).get(value, "")
                ).strip()
                alias = configured_alias or (
                    f"trace_{len(self._discovered_client_ips) + 1:03d}"
                )
                self._discovered_client_ips.append(value)
                self._discovered_client_aliases[value] = alias
                created = True

        if created:
            print(
                f"[proxy] discovered client {value} -> {alias} "
                f"[{source}]"
            )

        monitor = self._live_architecture_monitor
        if notify_monitor and monitor is not None and alias:
            try:
                monitor.register_client(value, alias=alias)
            except Exception as exc:
                print(
                    f"[proxy] WARNING: live monitor could not register "
                    f"client {value}: {type(exc).__name__}: {exc}"
                )
        return alias

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
                automatic_discovery = (
                    self._client_discovery_mode() == "automatic"
                )
                exclude_hosts = (
                    [str(self.config["proxy"]["upstream_host"])]
                    if automatic_discovery
                    else []
                )
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
                self._capture_preflight["client_discovery_mode"] = (
                    self._client_discovery_mode()
                )
                self._capture_preflight["capture_filter"] = build_capture_filter(
                    host=(
                        str(self.config["proxy"].get("listen_host", ""))
                        if (
                            automatic_discovery
                            and str(
                                self.config["proxy"].get("listen_host", "")
                            ).strip() not in {"", "0.0.0.0", "::"}
                        )
                        else None
                    ),
                    hosts=(
                        client_ips if not automatic_discovery else None
                    ),
                    port=int(self.config["proxy"]["listen_port"]),
                    exclude_hosts=exclude_hosts,
                )

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
                    host=(
                        str(self.config["proxy"].get("listen_host", ""))
                        if (
                            automatic_discovery
                            and str(
                                self.config["proxy"].get("listen_host", "")
                            ).strip() not in {"", "0.0.0.0", "::"}
                        )
                        else None
                    ),
                    hosts=(client_ips if not automatic_discovery else None),
                    exclude_hosts=exclude_hosts,
                    port=int(
                        self.config["proxy"]["listen_port"]
                    ),
                    snaplen_bytes=capture.get(
                        "snaplen_bytes", 256
                    ),
                )
                self._start_live_architecture_monitor(
                    interface=interface,
                    client_ips=client_ips,
                    exclude_hosts=exclude_hosts,
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
                if self._client_discovery_mode() == "automatic":
                    print(
                        "[proxy] participating-client discovery: automatic; "
                        f"excluding upstream {self.config['proxy']['upstream_host']} "
                        "from the capture filter"
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
                client_ip = (
                    str(client_addr[0]).strip()
                    if isinstance(client_addr, tuple) and client_addr
                    else str(client_addr).strip()
                )
                if self._client_discovery_mode() == "automatic":
                    self._register_discovered_client(
                        client_ip,
                        source="accepted_connection",
                    )
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

            self._stop_live_architecture_monitor()
            capture_result = self._stop_and_extract_capture()
            if self._live_architecture_summary:
                capture_result[
                    "live_architecture_inference"
                ] = self._live_architecture_summary
            self._restore_capture_offloads()

        summary = self._write_summary(capture_result)
        return summary

    def _start_live_architecture_monitor(
        self,
        *,
        interface: str,
        client_ips: list[str],
        exclude_hosts: list[str],
    ) -> None:
        inference = self.config.get(
            "architecture_inference", {}
        ) or {}
        if not (
            bool(inference.get("enabled", False))
            and bool(inference.get("realtime_enabled", True))
        ):
            return

        capture = self.config["capture"]
        window_sizes = (
            capture.get("window_sizes_sec")
            or [capture.get("window_seconds", 5.0)]
        )

        try:
            monitor = LiveArchitectureMonitor(
                experiment_id=self.experiment_id,
                interface=interface,
                client_ips=client_ips,
                client_aliases=(
                    capture.get("client_aliases", {}) or {}
                ),
                port=int(
                    self.config["proxy"]["listen_port"]
                ),
                proxy_ip=str(
                    self.config["proxy"].get("listen_host", "")
                ),
                exclude_hosts=exclude_hosts,
                output_dir=self.output_dir,
                window_sizes_sec=window_sizes,
                burst_gap_sec=float(
                    capture.get("burst_gap_sec", 0.05)
                ),
                idle_threshold_sec=float(
                    capture.get(
                        "idle_threshold_sec", 0.5
                    )
                ),
                model_root=str(
                    inference.get(
                        "model_root",
                        "fingerprinting_models",
                    )
                ),
                snaplen_bytes=capture.get(
                    "snaplen_bytes", 256
                ),
                feature_modes=tuple(
                    inference.get(
                        "feature_modes",
                        ["full", "size_normalized"],
                    )
                ),
                confidence_threshold=float(
                    inference.get(
                        "confidence_threshold", 0.90
                    )
                ),
                stable_windows=int(
                    inference.get("stable_windows", 3)
                ),
            )
            monitor.start()
            self._live_architecture_monitor = monitor

            if monitor.model_count:
                print(
                    "[proxy] real-time architecture inference ACTIVE: "
                    f"{monitor.model_count} model bundles loaded"
                )
            else:
                print(
                    "[proxy] real-time architecture feature collection "
                    "ACTIVE; no trained bundles found yet"
                )
        except Exception as exc:
            if bool(
                inference.get("realtime_required", True)
            ):
                raise ProxyError(
                    "Real-time architecture monitor failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            print(
                "[proxy] WARNING: real-time architecture monitor "
                f"unavailable: {type(exc).__name__}: {exc}"
            )

    def _stop_live_architecture_monitor(self) -> Dict[str, Any]:
        monitor = self._live_architecture_monitor
        if monitor is None:
            return self._live_architecture_summary
        try:
            self._live_architecture_summary = monitor.stop()
        except Exception as exc:
            self._live_architecture_summary = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            self._live_architecture_monitor = None
        return self._live_architecture_summary

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
                client_ips=self._capture_client_ips(),
                client_aliases=self._capture_client_aliases(),
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
                window_sizes_sec=(
                    capture.get("window_sizes_sec") or None
                ),
                capture_isolation_metadata={
                    "mode": (
                        "automatic_proxy_connection_discovery"
                        if self._client_discovery_mode() == "automatic"
                        else "manual_client_ip_bpf"
                    ),
                    "discovery_method": (
                        "accepted_proxy_connections"
                        if self._client_discovery_mode() == "automatic"
                        else "configured_client_ips"
                    ),
                    "upstream_server_excluded": str(
                        self.config["proxy"]["upstream_host"]
                    ),
                    "capture_filter_scope": "client_facing_only",
                },
            )
            result.update(artifacts)

            inference = self.config.get(
                "architecture_inference", {}
            ) or {}
            if (
                bool(inference.get("enabled", False))
                and bool(
                    inference.get("final_enabled", True)
                )
                and artifacts.get("per_client_artifacts")
            ):
                result[
                    "final_architecture_inference"
                ] = run_final_architecture_inference(
                    experiment_id=self.experiment_id,
                    per_client_artifacts=artifacts[
                        "per_client_artifacts"
                    ],
                    output_dir=self.output_dir,
                    model_root=str(
                        inference.get(
                            "model_root",
                            "fingerprinting_models",
                        )
                    ),
                    feature_modes=tuple(
                        inference.get(
                            "feature_modes",
                            ["full", "size_normalized"],
                        )
                    ),
                )
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
            "storage_locator": self.config.get("experiment", {}).get("storage_locator"),
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
                "client_discovery_mode": self._client_discovery_mode(),
                "client_ips_configured": len(
                    self._configured_client_ips()
                ),
                "client_ips_discovered": len(
                    self._capture_client_ips()
                ),
                "discovered_client_ips": self._capture_client_ips(),
                "client_aliases": self._capture_client_aliases(),
                "strict_client_isolation": bool(
                    self.config.get("capture", {}).get(
                        "strict_client_isolation", True
                    )
                ),
                "capture_filter_scope": "client_facing_only",
                "upstream_server_excluded": str(
                    self.config.get("proxy", {}).get("upstream_host", "")
                ),
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
