from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .architecture_models import (
    discover_bundle,
    load_bundle,
    predict_hierarchy,
)
from .capture import build_capture_filter
from .traffic.analysis import (
    PacketRecord,
    TSHARK_FIELDS,
    _first_nonempty,
    _parse_multi_int,
    _parse_tcp_flag_bits,
    _resolve_transport,
    _resolved_client_aliases,
    _to_float,
    _to_int,
    extract_feature_rows,
    resolve_direction,
)


class LiveInferenceError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _packet_from_columns(
    columns: Sequence[str],
    client_ips: Sequence[str],
) -> PacketRecord:
    values = list(columns)
    if len(values) < len(TSHARK_FIELDS):
        values.extend([""] * (len(TSHARK_FIELDS) - len(values)))

    (
        frame_number,
        timestamp_epoch,
        frame_length,
        ip_src,
        ipv6_src,
        ip_dst,
        ipv6_dst,
        tcp_srcport,
        udp_srcport,
        tcp_dstport,
        udp_dstport,
        ip_proto,
        ipv6_nxt,
        tcp_flags,
        tcp_syn,
        tcp_ack,
        tcp_fin,
        tcp_rst,
        tcp_retransmission,
        tcp_fast_retransmission,
        tls_record_length,
    ) = values[: len(TSHARK_FIELDS)]

    src_ip = _first_nonempty(ip_src, ipv6_src)
    dst_ip = _first_nonempty(ip_dst, ipv6_dst)
    src_port = _to_int(_first_nonempty(tcp_srcport, udp_srcport))
    dst_port = _to_int(_first_nonempty(tcp_dstport, udp_dstport))

    parsed_syn, parsed_ack, parsed_fin, parsed_rst = _parse_tcp_flag_bits(
        tcp_flags,
        syn_field=tcp_syn,
        ack_field=tcp_ack,
        fin_field=tcp_fin,
        rst_field=tcp_rst,
    )
    retransmission = int(
        bool((tcp_retransmission or "").strip())
        or bool((tcp_fast_retransmission or "").strip())
    )

    return PacketRecord(
        index=_to_int(frame_number),
        timestamp_epoch=_to_float(timestamp_epoch),
        frame_length=_to_int(frame_length),
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        transport_protocol=_resolve_transport(ip_proto, ipv6_nxt),
        tcp_flags_hex=tcp_flags or "",
        tcp_syn=parsed_syn,
        tcp_ack=parsed_ack,
        tcp_fin=parsed_fin,
        tcp_rst=parsed_rst,
        retransmission=retransmission,
        tls_record_lengths=_parse_multi_int(tls_record_length),
        direction=resolve_direction(
            src_ip=src_ip,
            dst_ip=dst_ip,
            client_ips=client_ips,
        ),
    )


class _TraceState:
    def __init__(
        self,
        alias: str,
        first_epoch: float,
        window_sizes: Sequence[float],
    ) -> None:
        self.alias = alias
        self.first_epoch = float(first_epoch)
        self.packets: Deque[PacketRecord] = deque()
        self.next_window_index = {
            float(size): 0 for size in window_sizes
        }


class LiveArchitectureMonitor:
    """Capture client-facing packet metadata and emit online predictions.

    This is a second, metadata-only tshark reader alongside the archival PCAP
    capture. It never terminates TLS, parses AI payloads, or consumes client/
    server ground truth. Client IPs are used only to group packets into traces.
    """

    def __init__(
        self,
        *,
        experiment_id: str,
        interface: str,
        client_ips: Sequence[str],
        client_aliases: Mapping[str, str] | None,
        port: int,
        proxy_ip: Optional[str] = None,
        exclude_hosts: Sequence[str] = (),
        output_dir: str | Path,
        window_sizes_sec: Sequence[float],
        burst_gap_sec: float,
        idle_threshold_sec: float,
        model_root: str | Path,
        snaplen_bytes: Optional[int] = 256,
        feature_modes: Sequence[str] = ("full", "size_normalized"),
        confidence_threshold: float = 0.90,
        stable_windows: int = 3,
    ) -> None:
        self.experiment_id = str(experiment_id)
        self.interface = str(interface)
        self.client_ips = [
            str(value).strip()
            for value in client_ips
            if str(value).strip()
        ]
        self.aliases = _resolved_client_aliases(
            self.client_ips,
            client_aliases=dict(client_aliases or {}),
        )
        self._configured_aliases = dict(client_aliases or {})
        self.proxy_ip = str(proxy_ip or "").strip() or None
        self.exclude_hosts = [
            str(value).strip()
            for value in exclude_hosts
            if str(value).strip()
        ]
        self._exclude_host_set = set(self.exclude_hosts)
        self.port = int(port)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.window_sizes = sorted(
            {
                float(value)
                for value in window_sizes_sec
                if float(value) > 0
            }
        )
        if not self.window_sizes:
            raise LiveInferenceError(
                "At least one positive real-time window size is required"
            )

        self.burst_gap_sec = float(burst_gap_sec)
        self.idle_threshold_sec = float(idle_threshold_sec)
        self.model_root = Path(model_root)
        self.snaplen_bytes = (
            int(snaplen_bytes)
            if snaplen_bytes is not None
            else None
        )
        self.feature_modes = tuple(feature_modes)
        self.confidence_threshold = float(confidence_threshold)
        self.stable_windows = max(1, int(stable_windows))

        self.feature_csv = (
            self.output_dir
            / f"{self.experiment_id}_live_features.csv"
        )
        self.prediction_jsonl = (
            self.output_dir
            / f"{self.experiment_id}_live_architecture_predictions.jsonl"
        )
        self.summary_json = (
            self.output_dir
            / f"{self.experiment_id}_live_architecture_summary.json"
        )

        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._ticker_thread: Optional[threading.Thread] = None
        self._traces: Dict[str, _TraceState] = {}
        # A few TCP handshake packets can reach tshark before accept() returns.
        # Buffer them briefly by peer and release them only after the proxy
        # confirms an actual accepted client connection.
        self._pending_packets: Dict[str, Deque[PacketRecord]] = {}
        self._connection_aliases: Dict[Tuple[str, int], str] = {}
        self._capture_start_epoch: Optional[float] = None
        self._feature_writer = None
        self._feature_handle = None
        self._prediction_handle = None
        self._models: Dict[Tuple[str, float], Dict[str, Any]] = {}
        self._stability: Dict[
            Tuple[str, str, float, str], List[str]
        ] = defaultdict(list)
        self._stable_first: Dict[
            Tuple[str, str, float, str], float
        ] = {}
        self._prediction_count = 0
        self._feature_count = 0
        self._load_models()

    def _load_models(self) -> None:
        for feature_mode in self.feature_modes:
            for size in self.window_sizes:
                path = discover_bundle(
                    self.model_root,
                    mode="realtime",
                    feature_mode=feature_mode,
                    window_size_sec=size,
                )
                if path is not None:
                    self._models[(feature_mode, size)] = load_bundle(path)

    @property
    def model_count(self) -> int:
        return len(self._models)

    def start(self) -> None:
        tshark = shutil.which("tshark")
        if not tshark:
            raise LiveInferenceError(
                "Real-time architecture inference requires tshark"
            )

        capture_filter = build_capture_filter(
            host=(
                self.proxy_ip
                if (
                    not self.client_ips
                    and self.proxy_ip not in {None, "", "0.0.0.0", "::"}
                )
                else None
            ),
            hosts=self.client_ips if self.client_ips else None,
            port=self.port,
            exclude_hosts=self.exclude_hosts,
        )

        command = [
            tshark,
            "-l",
            "-n",
            "-i",
            self.interface,
        ]
        if self.snaplen_bytes is not None and self.snaplen_bytes > 0:
            command.extend(["-s", str(self.snaplen_bytes)])
        if capture_filter:
            command.extend(["-f", capture_filter])
        command.extend(
            [
                "-Y",
                "ip or ipv6",
                "-T",
                "fields",
                "-E",
                "header=n",
                "-E",
                "separator=\t",
                "-E",
                "quote=n",
                "-E",
                "occurrence=a",
                "-E",
                "aggregator=;",
            ]
        )
        for field in TSHARK_FIELDS:
            command.extend(["-e", field])

        self._feature_handle = self.feature_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._prediction_handle = self.prediction_jsonl.open(
            "w",
            encoding="utf-8",
        )

        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if self._process.stdout is None:
            raise LiveInferenceError(
                "Unable to open tshark real-time output"
            )

        time.sleep(0.2)
        if self._process.poll() is not None:
            raise LiveInferenceError(
                "Real-time tshark process exited during startup"
            )

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="aifp-live-tshark-reader",
            daemon=True,
        )
        self._ticker_thread = threading.Thread(
            target=self._ticker_loop,
            name="aifp-live-window-ticker",
            daemon=True,
        )
        self._reader_thread.start()
        self._ticker_thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()

        process = self._process
        if process is not None and process.poll() is None:
            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGINT)
                process.wait(timeout=5)
            except Exception:
                process.kill()
                process.wait(timeout=2)

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3)
        if self._ticker_thread is not None:
            self._ticker_thread.join(timeout=3)

        # Flush any windows whose endpoint is at/before the final observed time.
        with self._lock:
            final_epoch = max(
                (
                    packet.timestamp_epoch
                    for state in self._traces.values()
                    for packet in state.packets
                ),
                default=time.time(),
            )
            self._emit_due_windows(final_epoch)

        if self._feature_handle is not None:
            self._feature_handle.close()
        if self._prediction_handle is not None:
            self._prediction_handle.close()

        summary = {
            "experiment_id": self.experiment_id,
            "status": "stopped",
            "window_sizes_sec": self.window_sizes,
            "model_count": self.model_count,
            "feature_modes": list(self.feature_modes),
            "feature_rows_emitted": self._feature_count,
            "predictions_emitted": self._prediction_count,
            "confidence_threshold": self.confidence_threshold,
            "stable_windows": self.stable_windows,
            "feature_csv": str(self.feature_csv),
            "prediction_jsonl": str(self.prediction_jsonl),
            "discovered_client_ips": list(self.client_ips),
            "client_aliases": dict(self.aliases),
            "unaccepted_peer_buffers_discarded": sum(
                len(values) for values in self._pending_packets.values()
            ),
            "stable_first": [
                {
                    "trace_id": key[0],
                    "feature_mode": key[1],
                    "window_size_sec": key[2],
                    "level": key[3],
                    "elapsed_sec": value,
                }
                for key, value in sorted(self._stable_first.items())
            ],
        }
        self.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    def register_connection(
        self,
        client_ip: str,
        client_port: int,
        *,
        alias: Optional[str] = None,
    ) -> str:
        value = str(client_ip).strip()
        port = int(client_port)
        if not value or port <= 0:
            raise LiveInferenceError("client_ip and client_port are required")
        key = (value, port)
        with self._lock:
            existing = self._connection_aliases.get(key)
            if existing:
                return existing
            selected = str(alias or "").strip() or f"trace_{len(self._connection_aliases)+1:03d}"
            self._connection_aliases[key] = selected
            if value not in self.client_ips:
                self.client_ips.append(value)
            pending_key = f"{value}:{port}"
            pending = list(self._pending_packets.pop(pending_key, ()))
            for packet in pending:
                packet = replace(packet, direction=resolve_direction(
                    src_ip=packet.src_ip, dst_ip=packet.dst_ip, client_ips=[value]
                ))
                self._append_registered_packet_locked(selected, packet)
            return selected

    def register_client(
        self,
        client_ip: str,
        *,
        alias: Optional[str] = None,
    ) -> str:
        """Register a client for live trace grouping.

        Registration may come from the proxy accept() path or from the
        metadata reader itself. The IP is grouping metadata only; it is never
        written into classifier-safe predictor rows.
        """
        value = str(client_ip).strip()
        if not value:
            raise LiveInferenceError("client_ip is required")

        with self._lock:
            existing = self.aliases.get(value)
            if existing:
                return existing

            selected = str(alias or "").strip()
            if not selected:
                selected = str(self._configured_aliases.get(value, "")).strip()
            if not selected:
                selected = f"trace_{len(self.aliases) + 1:03d}"

            if value not in self.client_ips:
                self.client_ips.append(value)
            self.aliases[value] = selected

            pending = list(self._pending_packets.pop(value, ()))
            for packet in pending:
                packet = replace(
                    packet,
                    direction=resolve_direction(
                        src_ip=packet.src_ip,
                        dst_ip=packet.dst_ip,
                        client_ips=self.client_ips,
                    ),
                )
                self._append_registered_packet_locked(selected, packet)
            return selected

    def _append_registered_packet_locked(
        self,
        alias: str,
        packet: PacketRecord,
    ) -> None:
        if self._capture_start_epoch is None:
            self._capture_start_epoch = packet.timestamp_epoch
        state = self._traces.get(alias)
        if state is None:
            state = _TraceState(
                alias,
                packet.timestamp_epoch,
                self.window_sizes,
            )
            self._traces[alias] = state
        state.packets.append(packet)

    def _infer_client_peer(self, packet: PacketRecord) -> Optional[Tuple[str, int]]:
        if not self.proxy_ip:
            return None
        if packet.src_ip == self.proxy_ip and packet.dst_ip:
            peer, peer_port = packet.dst_ip, packet.dst_port
        elif packet.dst_ip == self.proxy_ip and packet.src_ip:
            peer, peer_port = packet.src_ip, packet.src_port
        else:
            return None
        if peer in self._exclude_host_set or int(peer_port) <= 0:
            return None
        return peer, int(peer_port)

    def _reader_loop(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None

        for line in self._process.stdout:
            if self._stop.is_set():
                break
            if not line.strip():
                continue
            with self._lock:
                client_snapshot = list(self.client_ips)
                alias_snapshot = dict(self.aliases)
            packet = _packet_from_columns(
                line.rstrip("\n").split("\t"),
                client_snapshot,
            )
            peer = self._infer_client_peer(packet)
            if peer is None:
                continue
            with self._lock:
                alias = self._connection_aliases.get(peer)
            if alias is None:
                pending_key = f"{peer[0]}:{peer[1]}"
                with self._lock:
                    pending = self._pending_packets.get(pending_key)
                    if pending is None:
                        pending = deque(maxlen=256)
                        self._pending_packets[pending_key] = pending
                    pending.append(packet)
                continue
            with self._lock:
                self._append_registered_packet_locked(alias, packet)

    def _ticker_loop(self) -> None:
        while not self._stop.wait(0.10):
            now = time.time()
            with self._lock:
                self._emit_due_windows(now)

    def _emit_due_windows(self, now_epoch: float) -> None:
        for state in self._traces.values():
            for size in self.window_sizes:
                while True:
                    index = state.next_window_index[size]
                    start_epoch = state.first_epoch + index * size
                    end_epoch = start_epoch + size
                    if end_epoch > now_epoch:
                        break

                    packets = [
                        packet
                        for packet in state.packets
                        if start_epoch
                        <= packet.timestamp_epoch
                        < end_epoch
                    ]
                    self._emit_feature_window(
                        state,
                        size,
                        index,
                        start_epoch,
                        end_epoch,
                        packets,
                    )
                    state.next_window_index[size] += 1

            # Keep only packets that could still be needed by at least one
            # not-yet-emitted scale.
            earliest_needed = min(
                state.first_epoch
                + state.next_window_index[size] * size
                for size in self.window_sizes
            )
            while (
                state.packets
                and state.packets[0].timestamp_epoch < earliest_needed
            ):
                state.packets.popleft()

    def _emit_feature_window(
        self,
        state: _TraceState,
        size: float,
        index: int,
        start_epoch: float,
        end_epoch: float,
        packets: Sequence[PacketRecord],
    ) -> None:
        row = extract_feature_rows(
            packets=packets,
            experiment_id=self.experiment_id,
            burst_gap_sec=self.burst_gap_sec,
            idle_threshold_sec=self.idle_threshold_sec,
            window_seconds=None,
        )[0]
        start_local = index * size
        end_local = start_local + size
        global_offset = (
            state.first_epoch
            - (self._capture_start_epoch or state.first_epoch)
        )

        row.update(
            {
                "client_capture_id": state.alias,
                "row_type": "window",
                "window_index": index,
                "window_start_sec": start_local,
                "window_end_sec": end_local,
                "window_size_sec": size,
                "trace_start_offset_sec": global_offset,
                "window_start_global_sec": global_offset + start_local,
                "window_end_global_sec": global_offset + end_local,
            }
        )
        self._write_feature(row)
        self._feature_count += 1

        for feature_mode in self.feature_modes:
            bundle = self._models.get((feature_mode, size))
            if bundle is None:
                continue
            minimum_packets = int(
                bundle.get("realtime_min_packets") or 0
            )
            if int(row.get("packet_count_total", 0)) < minimum_packets:
                self._clear_stability(
                    state.alias,
                    feature_mode,
                    size,
                )
                continue
            prediction = predict_hierarchy(bundle, row)
            self._emit_prediction(
                state.alias,
                feature_mode,
                size,
                index,
                end_local,
                prediction,
            )

    def _write_feature(self, row: Mapping[str, Any]) -> None:
        if self._feature_writer is None:
            self._feature_writer = csv.DictWriter(
                self._feature_handle,
                fieldnames=list(row.keys()),
            )
            self._feature_writer.writeheader()
        self._feature_writer.writerow(dict(row))
        self._feature_handle.flush()

    def _clear_stability(
        self,
        trace_id: str,
        feature_mode: str,
        size: float,
    ) -> None:
        for level in ("family", "architecture", "variant", "application"):
            self._stability[
                (trace_id, feature_mode, size, level)
            ].clear()

    def _emit_prediction(
        self,
        trace_id: str,
        feature_mode: str,
        size: float,
        window_index: int,
        elapsed_sec: float,
        prediction: Mapping[str, Any],
    ) -> None:
        stable: Dict[str, Any] = {}
        for level in ("family", "architecture", "variant", "application"):
            level_result = prediction[level]
            label = level_result.get("label")
            confidence = float(level_result.get("confidence", 0.0))
            key = (trace_id, feature_mode, size, level)
            history = self._stability[key]

            accepted = (
                bool(label)
                and confidence >= self.confidence_threshold
            )
            if accepted:
                history.append(str(label))
                if len(history) > self.stable_windows:
                    del history[:-self.stable_windows]
            else:
                history.clear()

            is_stable = (
                len(history) >= self.stable_windows
                and len(set(history[-self.stable_windows:])) == 1
            )
            if is_stable and key not in self._stable_first:
                self._stable_first[key] = float(elapsed_sec)
            stable[level] = {
                "accepted": accepted,
                "stable": is_stable,
                "stable_label": (
                    history[-1] if is_stable else None
                ),
            }

        record = {
            "timestamp_utc": _utc_now_iso(),
            "experiment_id": self.experiment_id,
            "trace_id": trace_id,
            "feature_mode": feature_mode,
            "window_size_sec": size,
            "window_index": window_index,
            "elapsed_sec": elapsed_sec,
            "prediction": prediction,
            "gate": stable,
        }
        self._prediction_handle.write(
            json.dumps(record, sort_keys=True) + "\n"
        )
        self._prediction_handle.flush()
        self._prediction_count += 1

        architecture = prediction["architecture"]
        arch_label = architecture.get("label")
        arch_conf = float(architecture.get("confidence", 0.0))
        marker = (
            " STABLE"
            if stable["architecture"]["stable"]
            else ""
        )
        if arch_label:
            print(
                f"[architecture-live] {trace_id} "
                f"{feature_mode} {size:g}s "
                f"architecture={arch_label} "
                f"confidence={arch_conf:.3f}{marker}"
            )


def _read_overall_feature_row(path: str | Path) -> Optional[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("row_type", "")) == "overall":
                return row
    return None


def run_final_architecture_inference(
    *,
    experiment_id: str,
    per_client_artifacts: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    model_root: str | Path,
    feature_modes: Sequence[str] = ("full", "size_normalized"),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, Dict[str, Any]] = {}
    for feature_mode in feature_modes:
        path = discover_bundle(
            model_root,
            mode="final",
            feature_mode=feature_mode,
        )
        if path is not None:
            models[feature_mode] = load_bundle(path)

    result: Dict[str, Any] = {
        "experiment_id": str(experiment_id),
        "mode": "final",
        "model_root": str(model_root),
        "available_feature_modes": sorted(models),
        "traces": {},
    }

    for trace_id, artifacts in per_client_artifacts.items():
        feature_path = artifacts.get("features_csv")
        if not feature_path:
            continue
        row = _read_overall_feature_row(feature_path)
        if row is None:
            continue

        trace_result: Dict[str, Any] = {}
        for feature_mode in feature_modes:
            bundle = models.get(feature_mode)
            if bundle is None:
                trace_result[feature_mode] = {
                    "status": "model_unavailable"
                }
                continue
            prediction = predict_hierarchy(bundle, row)
            trace_result[feature_mode] = {
                "status": "predicted",
                "prediction": prediction,
            }
            architecture = prediction["architecture"]
            if architecture.get("label"):
                print(
                    f"[architecture-final] {trace_id} {feature_mode} "
                    f"architecture={architecture['label']} "
                    f"confidence={float(architecture['confidence']):.3f}"
                )
        result["traces"][trace_id] = trace_result

    target = (
        output_dir
        / f"{experiment_id}_final_architecture_predictions.json"
    )
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result["output_json"] = str(target)
    return result
