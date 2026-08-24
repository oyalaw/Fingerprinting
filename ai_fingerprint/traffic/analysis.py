from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..fingerprinting_dataset import sanitize_packet_sequence


FEATURE_SCHEMA_VERSION = "1.0"
PACKET_SCHEMA_VERSION = "1.0"


class FeatureExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PacketRecord:
    index: int
    timestamp_epoch: float
    frame_length: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    transport_protocol: str
    tcp_flags_hex: str
    tcp_syn: int
    tcp_ack: int
    tcp_fin: int
    tcp_rst: int
    retransmission: int
    tls_record_lengths: Tuple[int, ...]
    direction: str = "unknown"


@dataclass(frozen=True)
class BurstRecord:
    direction: str
    start_epoch: float
    end_epoch: float
    packet_count: int
    byte_count: int


TSHARK_FIELDS: Tuple[str, ...] = (
    "frame.number",
    "frame.time_epoch",
    "frame.len",
    "ip.src",
    "ipv6.src",
    "ip.dst",
    "ipv6.dst",
    "tcp.srcport",
    "udp.srcport",
    "tcp.dstport",
    "udp.dstport",
    "ip.proto",
    "ipv6.nxt",
    "tcp.flags",
    "tcp.flags.syn",
    "tcp.flags.ack",
    "tcp.flags.fin",
    "tcp.flags.reset",
    "tcp.analysis.retransmission",
    "tcp.analysis.fast_retransmission",
    "tls.record.length",
)


PACKET_CSV_FIELDS: Tuple[str, ...] = (
    "experiment_id",
    "packet_index",
    "timestamp_epoch",
    "relative_time_sec",
    "frame_length",
    "direction",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "transport_protocol",
    "tcp_flags_hex",
    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "retransmission",
    "tls_record_count",
    "tls_record_lengths",
)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _to_int(value: str, default: int = 0) -> int:
    value = (value or "").strip()
    if not value:
        return default
    try:
        return int(value, 0)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return default


def _to_float(value: str, default: float = 0.0) -> float:
    value = (value or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _parse_multi_int(value: str) -> Tuple[int, ...]:
    if not value:
        return tuple()
    normalized = value.replace(",", ";")
    result: List[int] = []
    for piece in normalized.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        parsed = _to_int(piece, default=-1)
        if parsed >= 0:
            result.append(parsed)
    return tuple(result)


def _resolve_transport(ip_proto: str, ipv6_next_header: str) -> str:
    protocol_number = _to_int(_first_nonempty(ip_proto, ipv6_next_header), -1)
    if protocol_number == 6:
        return "TCP"
    if protocol_number == 17:
        return "UDP"
    if protocol_number == 1:
        return "ICMP"
    if protocol_number == 58:
        return "ICMPV6"
    return "OTHER"


def resolve_direction(
    src_ip: str,
    dst_ip: str,
    server_ip: Optional[str] = None,
    client_ip: Optional[str] = None,
) -> str:
    server_ip = (server_ip or "").strip()
    client_ip = (client_ip or "").strip()

    if server_ip:
        if dst_ip == server_ip:
            return "up"
        if src_ip == server_ip:
            return "down"

    if client_ip:
        if src_ip == client_ip:
            return "up"
        if dst_ip == client_ip:
            return "down"

    return "unknown"


def find_tshark() -> str:
    executable = shutil.which("tshark")
    if not executable:
        raise FeatureExtractionError(
            "tshark is required for PCAP feature extraction but was not found "
            "on PATH. Install Wireshark or tshark on the proxy node."
        )
    return executable


def read_pcap_with_tshark(
    pcap_path: str | Path,
    server_ip: Optional[str] = None,
    client_ip: Optional[str] = None,
) -> List[PacketRecord]:
    pcap = Path(pcap_path)
    if not pcap.exists():
        raise FeatureExtractionError(f"PCAP file not found: {pcap}")

    tshark = find_tshark()
    command = [
        tshark,
        "-r",
        str(pcap),
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
    for field_name in TSHARK_FIELDS:
        command.extend(["-e", field_name])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise FeatureExtractionError(
            "tshark failed while reading the capture:\n"
            f"{completed.stderr.strip()}"
        )

    packets: List[PacketRecord] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue

        columns = line.split("\t")
        if len(columns) < len(TSHARK_FIELDS):
            columns.extend([""] * (len(TSHARK_FIELDS) - len(columns)))

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
        ) = columns[: len(TSHARK_FIELDS)]

        src_ip = _first_nonempty(ip_src, ipv6_src)
        dst_ip = _first_nonempty(ip_dst, ipv6_dst)
        src_port = _to_int(_first_nonempty(tcp_srcport, udp_srcport))
        dst_port = _to_int(_first_nonempty(tcp_dstport, udp_dstport))

        retransmission = int(
            bool((tcp_retransmission or "").strip())
            or bool((tcp_fast_retransmission or "").strip())
        )

        direction = resolve_direction(
            src_ip=src_ip,
            dst_ip=dst_ip,
            server_ip=server_ip,
            client_ip=client_ip,
        )

        packets.append(
            PacketRecord(
                index=_to_int(frame_number, len(packets) + 1),
                timestamp_epoch=_to_float(timestamp_epoch),
                frame_length=_to_int(frame_length),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                transport_protocol=_resolve_transport(ip_proto, ipv6_nxt),
                tcp_flags_hex=tcp_flags or "",
                tcp_syn=_to_int(tcp_syn),
                tcp_ack=_to_int(tcp_ack),
                tcp_fin=_to_int(tcp_fin),
                tcp_rst=_to_int(tcp_rst),
                retransmission=retransmission,
                tls_record_lengths=_parse_multi_int(tls_record_length),
                direction=direction,
            )
        )

    packets.sort(key=lambda packet: (packet.timestamp_epoch, packet.index))
    return packets


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _entropy(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0

    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 0.0

    unique_count = len(np.unique(finite))
    if unique_count <= 1:
        return 0.0

    bins = min(32, max(4, int(math.sqrt(finite.size))))
    counts, _ = np.histogram(finite, bins=bins)
    counts = counts[counts > 0]
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _skewness(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std == 0:
        return 0.0
    centered = (values - mean) / std
    return float(np.mean(centered ** 3))


def _kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std == 0:
        return 0.0
    centered = (values - mean) / std
    return float(np.mean(centered ** 4) - 3.0)


def _distribution_features(
    prefix: str,
    values: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_entropy": 0.0,
            f"{prefix}_skewness": 0.0,
            f"{prefix}_kurtosis": 0.0,
        }

    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_std": float(array.std(ddof=0)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_q25": _quantile(array, 0.25),
        f"{prefix}_q75": _quantile(array, 0.75),
        f"{prefix}_p90": _quantile(array, 0.90),
        f"{prefix}_p95": _quantile(array, 0.95),
        f"{prefix}_entropy": _entropy(array),
        f"{prefix}_skewness": _skewness(array),
        f"{prefix}_kurtosis": _kurtosis(array),
    }


def _compact_distribution_features(
    prefix: str,
    values: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std(ddof=0)),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
    }


def _build_bursts(
    packets: Sequence[PacketRecord],
    burst_gap_sec: float,
) -> List[BurstRecord]:
    if not packets:
        return []

    bursts: List[BurstRecord] = []
    current_direction = packets[0].direction
    current_start = packets[0].timestamp_epoch
    current_end = packets[0].timestamp_epoch
    current_packets = 1
    current_bytes = packets[0].frame_length

    for previous, packet in zip(packets, packets[1:]):
        gap = max(0.0, packet.timestamp_epoch - previous.timestamp_epoch)
        same_burst = (
            packet.direction == current_direction
            and gap <= burst_gap_sec
        )

        if same_burst:
            current_end = packet.timestamp_epoch
            current_packets += 1
            current_bytes += packet.frame_length
            continue

        bursts.append(
            BurstRecord(
                direction=current_direction,
                start_epoch=current_start,
                end_epoch=current_end,
                packet_count=current_packets,
                byte_count=current_bytes,
            )
        )

        current_direction = packet.direction
        current_start = packet.timestamp_epoch
        current_end = packet.timestamp_epoch
        current_packets = 1
        current_bytes = packet.frame_length

    bursts.append(
        BurstRecord(
            direction=current_direction,
            start_epoch=current_start,
            end_epoch=current_end,
            packet_count=current_packets,
            byte_count=current_bytes,
        )
    )
    return bursts


def _canonical_connection(packet: PacketRecord) -> Tuple[Any, ...]:
    left = (packet.src_ip, packet.src_port)
    right = (packet.dst_ip, packet.dst_port)
    endpoints = tuple(sorted((left, right)))
    return (
        packet.transport_protocol,
        endpoints[0],
        endpoints[1],
    )


def _direction_switch_count(packets: Sequence[PacketRecord]) -> int:
    directions = [
        packet.direction
        for packet in packets
        if packet.direction in {"up", "down"}
    ]
    return sum(
        1
        for previous, current in zip(directions, directions[1:])
        if current != previous
    )


def _extract_single_feature_row(
    packets: Sequence[PacketRecord],
    experiment_id: str,
    row_type: str,
    window_index: int,
    window_start_sec: float,
    window_end_sec: float,
    burst_gap_sec: float,
    idle_threshold_sec: float,
) -> Dict[str, Any]:
    packet_count = len(packets)
    frame_lengths = np.asarray(
        [packet.frame_length for packet in packets],
        dtype=np.float64,
    )

    up_packets = [
        packet for packet in packets if packet.direction == "up"
    ]
    down_packets = [
        packet for packet in packets if packet.direction == "down"
    ]
    unknown_packets = [
        packet for packet in packets if packet.direction == "unknown"
    ]

    up_lengths = np.asarray(
        [packet.frame_length for packet in up_packets],
        dtype=np.float64,
    )
    down_lengths = np.asarray(
        [packet.frame_length for packet in down_packets],
        dtype=np.float64,
    )

    timestamps = np.asarray(
        [packet.timestamp_epoch for packet in packets],
        dtype=np.float64,
    )
    iats = np.diff(timestamps) if timestamps.size >= 2 else np.asarray([], dtype=np.float64)
    iats = np.maximum(iats, 0.0)

    duration_sec = (
        float(timestamps[-1] - timestamps[0])
        if timestamps.size >= 2
        else 0.0
    )

    bytes_total = int(frame_lengths.sum()) if frame_lengths.size else 0
    bytes_up = int(up_lengths.sum()) if up_lengths.size else 0
    bytes_down = int(down_lengths.sum()) if down_lengths.size else 0
    bytes_unknown = bytes_total - bytes_up - bytes_down

    bursts = _build_bursts(packets, burst_gap_sec=burst_gap_sec)
    up_bursts = [burst for burst in bursts if burst.direction == "up"]
    down_bursts = [burst for burst in bursts if burst.direction == "down"]

    burst_bytes = np.asarray(
        [burst.byte_count for burst in bursts],
        dtype=np.float64,
    )
    burst_packets = np.asarray(
        [burst.packet_count for burst in bursts],
        dtype=np.float64,
    )
    burst_durations = np.asarray(
        [
            max(0.0, burst.end_epoch - burst.start_epoch)
            for burst in bursts
        ],
        dtype=np.float64,
    )
    burst_intervals = np.asarray(
        [
            max(0.0, current.start_epoch - previous.end_epoch)
            for previous, current in zip(bursts, bursts[1:])
        ],
        dtype=np.float64,
    )

    idle_gaps = iats[iats > idle_threshold_sec]

    tls_lengths = np.asarray(
        [
            length
            for packet in packets
            for length in packet.tls_record_lengths
        ],
        dtype=np.float64,
    )

    tcp_packets = [
        packet for packet in packets
        if packet.transport_protocol == "TCP"
    ]
    udp_packets = [
        packet for packet in packets
        if packet.transport_protocol == "UDP"
    ]

    connections = {
        _canonical_connection(packet)
        for packet in packets
        if packet.src_ip and packet.dst_ip
    }

    row: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "row_type": row_type,
        "window_index": window_index,
        "window_start_sec": window_start_sec,
        "window_end_sec": window_end_sec,
        "packet_count_total": packet_count,
        "packet_count_up": len(up_packets),
        "packet_count_down": len(down_packets),
        "packet_count_unknown": len(unknown_packets),
        "bytes_total": bytes_total,
        "bytes_up": bytes_up,
        "bytes_down": bytes_down,
        "bytes_unknown": bytes_unknown,
        "duration_sec": duration_sec,
        "upload_download_packet_ratio": _safe_divide(
            len(up_packets),
            len(down_packets),
        ),
        "upload_download_byte_ratio": _safe_divide(
            bytes_up,
            bytes_down,
        ),
        "upload_packet_fraction": _safe_divide(
            len(up_packets),
            packet_count,
        ),
        "download_packet_fraction": _safe_divide(
            len(down_packets),
            packet_count,
        ),
        "upload_byte_fraction": _safe_divide(
            bytes_up,
            bytes_total,
        ),
        "download_byte_fraction": _safe_divide(
            bytes_down,
            bytes_total,
        ),
        "packets_per_second": _safe_divide(
            packet_count,
            duration_sec,
        ),
        "bytes_per_second": _safe_divide(
            bytes_total,
            duration_sec,
        ),
        "upload_packets_per_second": _safe_divide(
            len(up_packets),
            duration_sec,
        ),
        "download_packets_per_second": _safe_divide(
            len(down_packets),
            duration_sec,
        ),
        "upload_bytes_per_second": _safe_divide(
            bytes_up,
            duration_sec,
        ),
        "download_bytes_per_second": _safe_divide(
            bytes_down,
            duration_sec,
        ),
        "direction_switch_count": _direction_switch_count(packets),
        "direction_switch_rate": _safe_divide(
            _direction_switch_count(packets),
            max(packet_count - 1, 0),
        ),
        "burst_count_total": len(bursts),
        "burst_count_up": len(up_bursts),
        "burst_count_down": len(down_bursts),
        "burst_frequency_per_second": _safe_divide(
            len(bursts),
            duration_sec,
        ),
        "idle_gap_count": int(idle_gaps.size),
        "idle_time_total_sec": float(idle_gaps.sum()) if idle_gaps.size else 0.0,
        "idle_time_mean_sec": float(idle_gaps.mean()) if idle_gaps.size else 0.0,
        "idle_time_max_sec": float(idle_gaps.max()) if idle_gaps.size else 0.0,
        "tcp_packet_count": len(tcp_packets),
        "udp_packet_count": len(udp_packets),
        "tcp_syn_count": sum(packet.tcp_syn for packet in packets),
        "tcp_ack_count": sum(packet.tcp_ack for packet in packets),
        "tcp_fin_count": sum(packet.tcp_fin for packet in packets),
        "tcp_rst_count": sum(packet.tcp_rst for packet in packets),
        "tcp_retransmission_count": sum(
            packet.retransmission for packet in packets
        ),
        "tls_record_count": int(tls_lengths.size),
        "connection_count": len(connections),
    }

    row.update(_distribution_features("packet_size", frame_lengths))
    row.update(_compact_distribution_features("upload_packet_size", up_lengths))
    row.update(_compact_distribution_features("download_packet_size", down_lengths))
    row.update(_distribution_features("iat_sec", iats))
    row.update(_compact_distribution_features("burst_bytes", burst_bytes))
    row.update(_compact_distribution_features("burst_packets", burst_packets))
    row.update(_compact_distribution_features("burst_duration_sec", burst_durations))
    row.update(_compact_distribution_features("burst_interval_sec", burst_intervals))
    row.update(_compact_distribution_features("tls_record_size", tls_lengths))

    return row


def extract_feature_rows(
    packets: Sequence[PacketRecord],
    experiment_id: str,
    burst_gap_sec: float = 0.05,
    idle_threshold_sec: float = 0.5,
    window_seconds: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if burst_gap_sec < 0:
        raise ValueError("burst_gap_sec must be nonnegative")
    if idle_threshold_sec < 0:
        raise ValueError("idle_threshold_sec must be nonnegative")
    if window_seconds is not None and window_seconds <= 0:
        raise ValueError("window_seconds must be positive when provided")

    if not packets:
        return [
            _extract_single_feature_row(
                packets=[],
                experiment_id=experiment_id,
                row_type="overall",
                window_index=-1,
                window_start_sec=0.0,
                window_end_sec=0.0,
                burst_gap_sec=burst_gap_sec,
                idle_threshold_sec=idle_threshold_sec,
            )
        ]

    first_timestamp = packets[0].timestamp_epoch
    total_duration = max(
        0.0,
        packets[-1].timestamp_epoch - first_timestamp,
    )

    rows = [
        _extract_single_feature_row(
            packets=packets,
            experiment_id=experiment_id,
            row_type="overall",
            window_index=-1,
            window_start_sec=0.0,
            window_end_sec=total_duration,
            burst_gap_sec=burst_gap_sec,
            idle_threshold_sec=idle_threshold_sec,
        )
    ]

    if window_seconds is None:
        return rows

    window_count = max(1, int(math.ceil(total_duration / window_seconds)))
    for window_index in range(window_count):
        start = window_index * window_seconds
        end = start + window_seconds
        selected = [
            packet
            for packet in packets
            if start
            <= packet.timestamp_epoch - first_timestamp
            < end
        ]
        rows.append(
            _extract_single_feature_row(
                packets=selected,
                experiment_id=experiment_id,
                row_type="window",
                window_index=window_index,
                window_start_sec=start,
                window_end_sec=end,
                burst_gap_sec=burst_gap_sec,
                idle_threshold_sec=idle_threshold_sec,
            )
        )

    return rows


def write_packet_sequence_csv(
    packets: Sequence[PacketRecord],
    experiment_id: str,
    output_path: str | Path,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    first_timestamp = (
        packets[0].timestamp_epoch
        if packets
        else 0.0
    )

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(PACKET_CSV_FIELDS),
        )
        writer.writeheader()

        for packet in packets:
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "packet_index": packet.index,
                    "timestamp_epoch": packet.timestamp_epoch,
                    "relative_time_sec": max(
                        0.0,
                        packet.timestamp_epoch - first_timestamp,
                    ),
                    "frame_length": packet.frame_length,
                    "direction": packet.direction,
                    "src_ip": packet.src_ip,
                    "dst_ip": packet.dst_ip,
                    "src_port": packet.src_port,
                    "dst_port": packet.dst_port,
                    "transport_protocol": packet.transport_protocol,
                    "tcp_flags_hex": packet.tcp_flags_hex,
                    "tcp_syn": packet.tcp_syn,
                    "tcp_ack": packet.tcp_ack,
                    "tcp_fin": packet.tcp_fin,
                    "tcp_rst": packet.tcp_rst,
                    "retransmission": packet.retransmission,
                    "tls_record_count": len(packet.tls_record_lengths),
                    "tls_record_lengths": ";".join(
                        str(length)
                        for length in packet.tls_record_lengths
                    ),
                }
            )

    return target


def write_feature_csv(
    rows: Sequence[Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise FeatureExtractionError(
            "No feature rows were supplied for export"
        )

    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    experiment_id: str,
    pcap_path: Path,
    packet_csv_path: Path,
    fingerprint_sequence_csv_path: Path,
    feature_csv_path: Path,
    manifest_path: Path,
    packets: Sequence[PacketRecord],
    feature_rows: Sequence[Dict[str, Any]],
    server_ip: Optional[str],
    client_ip: Optional[str],
    burst_gap_sec: float,
    idle_threshold_sec: float,
    window_seconds: Optional[float],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_id": experiment_id,
        "generated_at_utc": _utc_now_iso(),
        "schemas": {
            "packet_sequence_raw": PACKET_SCHEMA_VERSION,
            "packet_sequence_fingerprint_safe": "1.0",
            "handcrafted_features": FEATURE_SCHEMA_VERSION,
        },
        "capture": {
            "pcap_path": str(pcap_path),
            "pcap_size_bytes": pcap_path.stat().st_size,
            "pcap_sha256": _sha256(pcap_path),
            "packet_count": len(packets),
            "capture_start_epoch": (
                packets[0].timestamp_epoch if packets else None
            ),
            "capture_end_epoch": (
                packets[-1].timestamp_epoch if packets else None
            ),
        },
        "direction_reference": {
            "server_ip": server_ip,
            "client_ip": client_ip,
            "meaning": {
                "up": "client to server",
                "down": "server to client",
                "unknown": "direction could not be resolved",
            },
        },
        "feature_parameters": {
            "burst_gap_sec": burst_gap_sec,
            "idle_threshold_sec": idle_threshold_sec,
            "window_seconds": window_seconds,
        },
        "outputs": {
            "packet_sequence_raw_csv": str(packet_csv_path),
            "packet_sequence_fingerprint_csv": str(
                fingerprint_sequence_csv_path
            ),
            "features_csv": str(feature_csv_path),
            "feature_row_count": len(feature_rows),
        },
        "extractor": {
            "parser": "tshark",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "fingerprinting_policy": {
            "predictor_source": "proxy_only",
            "contains_ai_ground_truth_labels": False,
            "contains_client_server_resource_telemetry": False,
            "raw_packet_sequence_classifier_eligible": False,
            "fingerprint_sequence_classifier_eligible": True,
            "raw_sequence_identity_fields_excluded_from_safe_sequence": [
                "timestamp_epoch",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
            ],
            "note": (
                "Only proxy-observable traffic may enter attacker feature "
                "sets. Client/server labels and resource telemetry are "
                "excluded. The raw packet CSV is retained for audit, while "
                "the fingerprint-safe sequence removes endpoint identity "
                "and absolute timestamp fields."
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def extract_capture_artifacts(
    pcap_path: str | Path,
    experiment_id: str,
    output_dir: str | Path | None = None,
    server_ip: Optional[str] = None,
    client_ip: Optional[str] = None,
    burst_gap_sec: float = 0.05,
    idle_threshold_sec: float = 0.5,
    window_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    pcap = Path(pcap_path)
    if not pcap.exists():
        raise FeatureExtractionError(
            f"PCAP file not found: {pcap}"
        )

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else pcap.parent
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    packets = read_pcap_with_tshark(
        pcap_path=pcap,
        server_ip=server_ip,
        client_ip=client_ip,
    )

    packet_csv = target_dir / f"{experiment_id}_packet_sequence.csv"
    fingerprint_sequence_csv = (
        target_dir
        / f"{experiment_id}_fingerprint_sequence.csv"
    )
    feature_csv = target_dir / f"{experiment_id}_features.csv"
    manifest_json = target_dir / f"{experiment_id}_manifest.json"

    feature_rows = extract_feature_rows(
        packets=packets,
        experiment_id=experiment_id,
        burst_gap_sec=burst_gap_sec,
        idle_threshold_sec=idle_threshold_sec,
        window_seconds=window_seconds,
    )

    write_packet_sequence_csv(
        packets=packets,
        experiment_id=experiment_id,
        output_path=packet_csv,
    )
    sanitize_packet_sequence(
        raw_packet_csv=packet_csv,
        output_csv=fingerprint_sequence_csv,
    )
    write_feature_csv(
        rows=feature_rows,
        output_path=feature_csv,
    )
    write_manifest(
        experiment_id=experiment_id,
        pcap_path=pcap,
        packet_csv_path=packet_csv,
        fingerprint_sequence_csv_path=fingerprint_sequence_csv,
        feature_csv_path=feature_csv,
        manifest_path=manifest_json,
        packets=packets,
        feature_rows=feature_rows,
        server_ip=server_ip,
        client_ip=client_ip,
        burst_gap_sec=burst_gap_sec,
        idle_threshold_sec=idle_threshold_sec,
        window_seconds=window_seconds,
    )

    overall = feature_rows[0]
    return {
        "experiment_id": experiment_id,
        "pcap": str(pcap),
        "packet_sequence_csv": str(packet_csv),
        "fingerprint_sequence_csv": str(fingerprint_sequence_csv),
        "features_csv": str(feature_csv),
        "manifest_json": str(manifest_json),
        "packet_count": len(packets),
        "feature_count": len(overall) - 6,
        "feature_row_count": len(feature_rows),
    }
