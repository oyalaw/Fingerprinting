from __future__ import annotations

"""Packet-budget feature extraction for cross-vantage AI fingerprinting.

The module intentionally uses only network-observable metadata.  Identifiers,
IP addresses, ports, payload contents, and ground-truth labels are never
predictors.
"""

import ast
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


PACKET_COUNTS = (1, 5, 10, 25, 50, 100, 250, 500)

METADATA_FIELDS = {
    "sample_id",
    "experiment_id",
    "client_id",
    "source_role",
    "source_participant",
    "source_file",
    "packet_budget",
    "block_index",
    "packet_start_index",
    "packet_end_index",
    "family",
    "architecture",
    "variant",
    "application",
}


@dataclass(frozen=True)
class PacketObservation:
    timestamp: float
    frame_length: float
    direction: str
    transport_protocol: str
    tcp_syn: int
    tcp_ack: int
    tcp_fin: int
    tcp_rst: int
    retransmission: int
    tls_record_lengths: tuple[float, ...]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def normalize_client_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def canonical_client_label(value: Any) -> str:
    raw = str(value or "").strip()
    norm = normalize_client_id(raw)
    match = re.search(r"client(\d+)", norm)
    if match:
        return f"client_{match.group(1)}"
    return raw or norm or "unknown"


def normalize_direction(value: Any, source_role: str) -> str:
    """Normalize direction relative to the client.

    Canonical semantics:
      up   = client -> server
      down = server -> client

    Endpoint-local inbound/outbound labels are reversed at the server.
    Proxy data should already be canonical up/down; inbound/outbound at a
    proxy is considered ambiguous and is mapped to unknown.
    """
    raw = str(value or "").strip().lower()
    role = str(source_role or "").strip().lower()
    if raw in {"up", "upload", "client_to_server", "c2s"}:
        return "up"
    if raw in {"down", "download", "server_to_client", "s2c"}:
        return "down"
    if role == "client":
        if raw in {"out", "outbound", "tx", "sent", "send"}:
            return "up"
        if raw in {"in", "inbound", "rx", "received", "recv"}:
            return "down"
    if role == "server":
        if raw in {"in", "inbound", "rx", "received", "recv"}:
            return "up"
        if raw in {"out", "outbound", "tx", "sent", "send"}:
            return "down"
    return "unknown"


def parse_tls_lengths(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(_to_float(x) for x in value if _to_float(x) >= 0)
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return tuple(_to_float(x) for x in parsed if _to_float(x) >= 0)
    except Exception:
        pass
    parts = [p for p in re.split(r"[,;|\s]+", text) if p]
    return tuple(_to_float(p) for p in parts if _to_float(p) >= 0)


def _timestamp_from_row(row: dict[str, Any]) -> float:
    for field in (
        "timestamp_epoch",
        "packet_timestamp",
        "timestamp",
        "time_epoch",
        "relative_time_sec",
        "relative_time",
    ):
        if field in row and str(row.get(field, "")).strip() != "":
            return _to_float(row.get(field))
    return 0.0


def rows_to_packets(
    rows: Iterable[dict[str, Any]],
    source_role: str,
) -> list[PacketObservation]:
    packets: list[PacketObservation] = []
    for row in rows:
        direction = normalize_direction(row.get("direction"), source_role)
        packets.append(
            PacketObservation(
                timestamp=_timestamp_from_row(row),
                frame_length=max(0.0, _to_float(row.get("frame_length"))),
                direction=direction,
                transport_protocol=str(
                    row.get("transport_protocol", row.get("protocol", "OTHER")) or "OTHER"
                ).strip().upper(),
                tcp_syn=int(bool(_to_int(row.get("tcp_syn")))),
                tcp_ack=int(bool(_to_int(row.get("tcp_ack")))),
                tcp_fin=int(bool(_to_int(row.get("tcp_fin")))),
                tcp_rst=int(bool(_to_int(row.get("tcp_rst")))),
                retransmission=int(bool(_to_int(row.get("retransmission")))),
                tls_record_lengths=parse_tls_lengths(row.get("tls_record_lengths")),
            )
        )
    packets.sort(key=lambda p: p.timestamp)
    return packets


def read_safe_sequence(path: Path, source_role: str) -> list[PacketObservation]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return rows_to_packets(csv.DictReader(handle), source_role)


def is_safe_sequence_csv(path: Path) -> bool:
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
    except Exception:
        return False
    fields = {str(x).strip() for x in header}
    has_time = bool(
        fields
        & {
            "timestamp_epoch",
            "packet_timestamp",
            "timestamp",
            "time_epoch",
            "relative_time_sec",
            "relative_time",
        }
    )
    return "frame_length" in fields and "direction" in fields and has_time


def infer_source_role(participant: str) -> str:
    value = str(participant or "").strip().lower()
    if value == "proxy" or value.startswith("proxy_"):
        return "proxy"
    if value == "server" or value.startswith("server_"):
        return "server"
    if value.startswith("client"):
        return "client"
    return "unknown"


def infer_client_ids_from_file(path: Path, participant: str) -> list[str]:
    candidates = [participant, path.stem, *[p.name for p in path.parents][:4]]
    found: list[str] = []
    for item in candidates:
        for match in re.finditer(r"client[_-]?(\d+)", str(item), flags=re.I):
            label = f"client_{match.group(1)}"
            if label not in found:
                found.append(label)
    return found


def read_sequence_groups(
    path: Path,
    source_role: str,
    participant: str,
) -> dict[str, list[PacketObservation]]:
    """Read a sequence CSV and split it by client when client metadata exists.

    Server-side files that mix multiple clients must contain a usable client
    column.  A server file with no client identifier is intentionally rejected
    by returning an empty mapping; mixing clients would invalidate the design.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    client_field = next(
        (
            name
            for name in (
                "resolved_client_id",
                "client_capture_id",
                "client_id",
                "client",
            )
            if name in fields
        ),
        None,
    )

    if client_field:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            client = canonical_client_label(row.get(client_field))
            if client and client != "unknown":
                grouped.setdefault(client, []).append(row)
        return {
            client: rows_to_packets(client_rows, source_role)
            for client, client_rows in grouped.items()
            if client_rows
        }

    inferred = infer_client_ids_from_file(path, participant)
    if len(inferred) == 1:
        return {inferred[0]: rows_to_packets(rows, source_role)}

    # Client participant directories are themselves a reliable identity.
    if source_role == "client":
        client = canonical_client_label(participant)
        if client != "unknown":
            return {client: rows_to_packets(rows, source_role)}

    # A server or proxy file without a client identity must not be mixed.
    return {}


def _distribution(prefix: str, values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_q75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std(ddof=0)),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_q25": float(np.quantile(arr, 0.25)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
    }


def _tls_summary(packet: PacketObservation) -> tuple[int, float, float, float]:
    lengths = packet.tls_record_lengths
    if not lengths:
        return 0, 0.0, 0.0, 0.0
    arr = np.asarray(lengths, dtype=np.float64)
    return int(arr.size), float(arr.sum()), float(arr.mean()), float(arr.max())


def single_packet_features(
    packet: PacketObservation,
    previous_timestamp: float | None,
) -> dict[str, float]:
    iat = max(0.0, packet.timestamp - previous_timestamp) if previous_timestamp is not None else 0.0
    tls_count, tls_sum, tls_mean, tls_max = _tls_summary(packet)
    return {
        "frame_length": float(packet.frame_length),
        "direction_up": float(packet.direction == "up"),
        "direction_down": float(packet.direction == "down"),
        "direction_unknown": float(packet.direction == "unknown"),
        "iat_sec": float(iat),
        "transport_tcp": float(packet.transport_protocol == "TCP"),
        "transport_udp": float(packet.transport_protocol == "UDP"),
        "tcp_syn": float(packet.tcp_syn),
        "tcp_ack": float(packet.tcp_ack),
        "tcp_fin": float(packet.tcp_fin),
        "tcp_rst": float(packet.tcp_rst),
        "retransmission": float(packet.retransmission),
        "tls_record_count": float(tls_count),
        "tls_record_size_sum": float(tls_sum),
        "tls_record_size_mean": float(tls_mean),
        "tls_record_size_max": float(tls_max),
    }


def packet_block_features(packets: Sequence[PacketObservation]) -> dict[str, float]:
    if not packets:
        raise ValueError("packet block must be non-empty")
    lengths = np.asarray([p.frame_length for p in packets], dtype=np.float64)
    timestamps = np.asarray([p.timestamp for p in packets], dtype=np.float64)
    iats = np.maximum(np.diff(timestamps), 0.0) if len(packets) >= 2 else np.asarray([], dtype=np.float64)
    span = float(max(0.0, timestamps[-1] - timestamps[0])) if len(packets) >= 2 else 0.0

    directions = [p.direction for p in packets]
    up = sum(d == "up" for d in directions)
    down = sum(d == "down" for d in directions)
    unknown = len(packets) - up - down
    valid_dirs = [d for d in directions if d in {"up", "down"}]
    switches = sum(a != b for a, b in zip(valid_dirs, valid_dirs[1:]))

    tls_lengths = [x for p in packets for x in p.tls_record_lengths]
    tls_records = sum(len(p.tls_record_lengths) for p in packets)

    result: dict[str, float] = {
        "elapsed_span_sec": span,
        "upload_packet_fraction": _safe_div(up, len(packets)),
        "download_packet_fraction": _safe_div(down, len(packets)),
        "unknown_packet_fraction": _safe_div(unknown, len(packets)),
        "upload_download_packet_ratio": _safe_div(up, down),
        "direction_switch_rate": _safe_div(switches, max(len(valid_dirs) - 1, 0)),
        "packets_per_second": _safe_div(len(packets), span),
        "bytes_per_second": _safe_div(float(lengths.sum()), span),
        "tcp_fraction": _safe_div(sum(p.transport_protocol == "TCP" for p in packets), len(packets)),
        "udp_fraction": _safe_div(sum(p.transport_protocol == "UDP" for p in packets), len(packets)),
        "tcp_syn_fraction": _safe_div(sum(p.tcp_syn for p in packets), len(packets)),
        "tcp_ack_fraction": _safe_div(sum(p.tcp_ack for p in packets), len(packets)),
        "tcp_fin_fraction": _safe_div(sum(p.tcp_fin for p in packets), len(packets)),
        "tcp_rst_fraction": _safe_div(sum(p.tcp_rst for p in packets), len(packets)),
        "retransmission_fraction": _safe_div(sum(p.retransmission for p in packets), len(packets)),
        "tls_records_per_packet": _safe_div(tls_records, len(packets)),
    }
    result.update(_distribution("packet_size", lengths.tolist()))
    result.update(_distribution("iat_sec", iats.tolist()))
    result.update(_distribution("tls_record_size", tls_lengths))
    return result


def iter_packet_budget_samples(
    packets: Sequence[PacketObservation],
    packet_budget: int,
) -> Iterator[tuple[int, int, int, dict[str, float]]]:
    """Yield non-overlapping packet-budget samples.

    Returns (block_index, start_index, end_index_inclusive, feature_dict).
    For K=1 every packet is evaluated individually.
    Incomplete final blocks are deliberately omitted so every K-packet sample
    contains exactly K packets.
    """
    k = int(packet_budget)
    if k <= 0:
        raise ValueError("packet_budget must be positive")
    if k == 1:
        previous: float | None = None
        for index, packet in enumerate(packets):
            yield index, index, index, single_packet_features(packet, previous)
            previous = packet.timestamp
        return
    block_index = 0
    for start in range(0, len(packets) - k + 1, k):
        end = start + k
        block = packets[start:end]
        yield block_index, start, end - 1, packet_block_features(block)
        block_index += 1
