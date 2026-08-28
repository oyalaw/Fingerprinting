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

from ..fingerprinting_dataset import SAFE_SEQUENCE_FIELDS, sanitize_packet_sequence


FEATURE_SCHEMA_VERSION = "1.2"
PACKET_SCHEMA_VERSION = "1.1"


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


def _normalized_client_ips(
    client_ip: Optional[str] = None,
    client_ips: Optional[Sequence[str]] = None,
) -> List[str]:
    values: List[str] = []
    if client_ip and str(client_ip).strip():
        values.append(str(client_ip).strip())
    if client_ips:
        values.extend(
            str(value).strip()
            for value in client_ips
            if str(value).strip()
        )
    return list(dict.fromkeys(values))


def _parse_tcp_flag_bits(
    tcp_flags_hex: str,
    syn_field: str = "",
    ack_field: str = "",
    fin_field: str = "",
    rst_field: str = "",
) -> Tuple[int, int, int, int]:
    """Return SYN/ACK/FIN/RST from the authoritative TCP flags bitmask.

    Some tshark builds emit blank/zero ``tcp.flags.syn`` style fields even
    when ``tcp.flags`` is present. The bitmask is therefore the primary
    source; individual tshark fields are used only when the bitmask is absent.
    """
    raw = (tcp_flags_hex or "").strip()
    if raw:
        # occurrence=a can theoretically aggregate values; a packet should
        # have one TCP header, so use the first parsable flag value.
        for piece in raw.replace(",", ";").split(";"):
            piece = piece.strip()
            if not piece:
                continue
            try:
                bits = int(piece, 0)
            except ValueError:
                continue
            return (
                int(bool(bits & 0x02)),  # SYN
                int(bool(bits & 0x10)),  # ACK
                int(bool(bits & 0x01)),  # FIN
                int(bool(bits & 0x04)),  # RST
            )

    return (
        int(bool(_to_int(syn_field))),
        int(bool(_to_int(ack_field))),
        int(bool(_to_int(fin_field))),
        int(bool(_to_int(rst_field))),
    )


def _packet_matches_clients(
    src_ip: str,
    dst_ip: str,
    client_ips: Sequence[str],
) -> bool:
    if not client_ips:
        return True
    allowed = set(client_ips)
    return src_ip in allowed or dst_ip in allowed


def _connection_key(ip: str, port: int) -> str:
    return f"{str(ip).strip()}:{int(port)}"


def connection_facing_packets(
    packets: Sequence[PacketRecord],
    *,
    client_ip: str,
    client_port: int,
    proxy_ip: Optional[str] = None,
    proxy_port: Optional[int] = None,
) -> List[PacketRecord]:
    """Return only packets from one accepted client TCP connection."""
    ip = str(client_ip).strip()
    port = int(client_port)
    proxy = str(proxy_ip or "").strip()
    pport = int(proxy_port) if proxy_port is not None else None
    result: List[PacketRecord] = []
    for packet in packets:
        up = packet.src_ip == ip and packet.src_port == port
        down = packet.dst_ip == ip and packet.dst_port == port
        if not (up or down):
            continue
        if proxy:
            peer_ip = packet.dst_ip if up else packet.src_ip
            if peer_ip != proxy:
                continue
        if pport is not None:
            peer_port = packet.dst_port if up else packet.src_port
            if peer_port != pport:
                continue
        result.append(packet)
    return result


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
    client_ips: Optional[Sequence[str]] = None,
) -> str:
    server_ip = (server_ip or "").strip()
    clients = _normalized_client_ips(
        client_ip=client_ip,
        client_ips=client_ips,
    )
    client_set = set(clients)

    # Client membership is the safest direction reference on an inline proxy.
    # It remains correct even when the proxy and upstream server use the same
    # TCP port, and it avoids inverting the upstream proxy-server leg.
    if client_set:
        if src_ip in client_set and dst_ip not in client_set:
            return "up"
        if dst_ip in client_set and src_ip not in client_set:
            return "down"

    if server_ip:
        if dst_ip == server_ip:
            return "up"
        if src_ip == server_ip:
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
    client_ips: Optional[Sequence[str]] = None,
    isolate_client_facing: bool = True,
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

    clients = _normalized_client_ips(
        client_ip=client_ip,
        client_ips=client_ips,
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

        if (
            isolate_client_facing
            and clients
            and not _packet_matches_clients(
                src_ip, dst_ip, clients
            )
        ):
            # Exclude the proxy-to-upstream duplicate byte stream.
            continue

        parsed_syn, parsed_ack, parsed_fin, parsed_rst = (
            _parse_tcp_flag_bits(
                tcp_flags,
                syn_field=tcp_syn,
                ack_field=tcp_ack,
                fin_field=tcp_fin,
                rst_field=tcp_rst,
            )
        )

        retransmission = int(
            bool((tcp_retransmission or "").strip())
            or bool((tcp_fast_retransmission or "").strip())
        )

        direction = resolve_direction(
            src_ip=src_ip,
            dst_ip=dst_ip,
            server_ip=server_ip,
            client_ip=client_ip,
            client_ips=clients,
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
                tcp_syn=parsed_syn,
                tcp_ack=parsed_ack,
                tcp_fin=parsed_fin,
                tcp_rst=parsed_rst,
                retransmission=retransmission,
                tls_record_lengths=_parse_multi_int(tls_record_length),
                direction=direction,
            )
        )

    packets.sort(key=lambda packet: (packet.timestamp_epoch, packet.index))
    return packets



def read_packet_sequence_csv(
    csv_path: str | Path,
    client_ip: Optional[str] = None,
    client_ips: Optional[Sequence[str]] = None,
    server_ip: Optional[str] = None,
    isolate_client_facing: bool = True,
) -> List[PacketRecord]:
    """Read an existing raw packet-sequence CSV and normalize it.

    This is intentionally independent of the original PCAP. It allows a run
    captured with an overly broad proxy filter to be repaired by specifying
    the participating client IPs: the upstream proxy-server duplicate leg is
    removed, direction is recomputed, and TCP flags are reconstructed from
    ``tcp_flags_hex``.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FeatureExtractionError(
            f"Packet sequence CSV not found: {path}"
        )

    clients = _normalized_client_ips(
        client_ip=client_ip,
        client_ips=client_ips,
    )
    packets: List[PacketRecord] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "packet_index",
            "timestamp_epoch",
            "frame_length",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "transport_protocol",
            "tcp_flags_hex",
            "retransmission",
            "tls_record_lengths",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise FeatureExtractionError(
                f"Raw packet sequence is missing columns: {missing}"
            )

        for row in reader:
            src_ip = str(row.get("src_ip", "")).strip()
            dst_ip = str(row.get("dst_ip", "")).strip()

            if (
                isolate_client_facing
                and clients
                and not _packet_matches_clients(
                    src_ip, dst_ip, clients
                )
            ):
                continue

            flags_hex = str(row.get("tcp_flags_hex", "") or "")
            syn, ack, fin, rst = _parse_tcp_flag_bits(
                flags_hex,
                syn_field=str(row.get("tcp_syn", "") or ""),
                ack_field=str(row.get("tcp_ack", "") or ""),
                fin_field=str(row.get("tcp_fin", "") or ""),
                rst_field=str(row.get("tcp_rst", "") or ""),
            )

            direction = resolve_direction(
                src_ip=src_ip,
                dst_ip=dst_ip,
                server_ip=server_ip,
                client_ips=clients,
            )

            packets.append(
                PacketRecord(
                    index=_to_int(str(row.get("packet_index", ""))),
                    timestamp_epoch=_to_float(
                        str(row.get("timestamp_epoch", ""))
                    ),
                    frame_length=_to_int(
                        str(row.get("frame_length", ""))
                    ),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=_to_int(
                        str(row.get("src_port", ""))
                    ),
                    dst_port=_to_int(
                        str(row.get("dst_port", ""))
                    ),
                    transport_protocol=str(
                        row.get("transport_protocol", "OTHER")
                    ).strip() or "OTHER",
                    tcp_flags_hex=flags_hex,
                    tcp_syn=syn,
                    tcp_ack=ack,
                    tcp_fin=fin,
                    tcp_rst=rst,
                    retransmission=int(
                        bool(_to_int(str(row.get("retransmission", "0"))))
                    ),
                    tls_record_lengths=_parse_multi_int(
                        str(row.get("tls_record_lengths", "") or "")
                    ),
                    direction=direction,
                )
            )

    packets.sort(key=lambda packet: (packet.timestamp_epoch, packet.index))
    return packets


def client_facing_packets(
    packets: Sequence[PacketRecord],
    client_ip: str,
) -> List[PacketRecord]:
    """Return one client's packets with direction recomputed for that client."""
    target = str(client_ip).strip()
    selected: List[PacketRecord] = []
    for packet in packets:
        if packet.src_ip != target and packet.dst_ip != target:
            continue
        selected.append(
            PacketRecord(
                index=packet.index,
                timestamp_epoch=packet.timestamp_epoch,
                frame_length=packet.frame_length,
                src_ip=packet.src_ip,
                dst_ip=packet.dst_ip,
                src_port=packet.src_port,
                dst_port=packet.dst_port,
                transport_protocol=packet.transport_protocol,
                tcp_flags_hex=packet.tcp_flags_hex,
                tcp_syn=packet.tcp_syn,
                tcp_ack=packet.tcp_ack,
                tcp_fin=packet.tcp_fin,
                tcp_rst=packet.tcp_rst,
                retransmission=packet.retransmission,
                tls_record_lengths=packet.tls_record_lengths,
                direction=resolve_direction(
                    src_ip=packet.src_ip,
                    dst_ip=packet.dst_ip,
                    client_ip=target,
                ),
            )
        )
    return selected


def capture_quality_diagnostics(
    packets: Sequence[PacketRecord],
    interface_mtu: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize packet-size artifacts that may reflect Linux offloading."""
    mtu = int(interface_mtu) if interface_mtu else None
    # Ethernet L2 headers/VLAN tags add a small amount beyond the IP MTU.
    # A 64-byte allowance is conservative and avoids flagging ordinary frames.
    threshold = (mtu + 64) if mtu else 2048
    lengths = [packet.frame_length for packet in packets]
    oversized = [value for value in lengths if value > threshold]
    maximum = max(lengths) if lengths else 0
    ratio = (
        float(len(oversized)) / float(len(lengths))
        if lengths
        else 0.0
    )
    return {
        "interface_mtu": mtu,
        "oversized_frame_threshold_bytes": threshold,
        "oversized_frame_count": len(oversized),
        "oversized_frame_fraction": ratio,
        "max_frame_length_bytes": maximum,
        "possible_offload_coalescing": bool(oversized),
        "note": (
            "Frames above the interface MTU allowance can be produced by "
            "GRO/GSO/TSO/LRO in the host capture stack and may not represent "
            "on-wire packet sizes. This diagnostic is metadata only and is "
            "not a classifier predictor."
        ),
    }

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
    packet_information_threshold: int = 2,
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
        "window_size_sec": 0.0,
        "packet_count_total": packet_count,
        # Sparse windows are retained rather than dropped. These audit fields
        # describe whether a window contains enough packet information for
        # downstream quality analysis and are excluded from predictor X.
        "packet_information_threshold": int(packet_information_threshold),
        "packet_information_ok": int(packet_count >= int(packet_information_threshold)),
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
    packet_information_threshold: int = 2,
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
                packet_information_threshold=packet_information_threshold,
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
            packet_information_threshold=packet_information_threshold,
        )
    ]

    if window_seconds is None:
        return rows

    window_count = max(1, int(math.ceil(total_duration / window_seconds)))

    # Linear-time binning. The earlier implementation scanned every packet
    # for every window, which becomes prohibitively expensive for millions
    # of packets at sub-second resolutions.
    bins: List[List[PacketRecord]] = [
        [] for _ in range(window_count)
    ]
    for packet in packets:
        relative = max(
            0.0,
            packet.timestamp_epoch - first_timestamp,
        )
        index = int(relative // window_seconds)
        if index >= window_count:
            # A packet exactly on the final endpoint belongs to the final
            # available bin rather than creating an unreported extra window.
            index = window_count - 1
        bins[index].append(packet)

    for window_index, selected in enumerate(bins):
        start = window_index * window_seconds
        end = start + window_seconds
        row = _extract_single_feature_row(
            packets=selected,
            experiment_id=experiment_id,
            row_type="window",
            window_index=window_index,
            window_start_sec=start,
            window_end_sec=end,
            burst_gap_sec=burst_gap_sec,
            idle_threshold_sec=idle_threshold_sec,
            packet_information_threshold=packet_information_threshold,
        )
        row["window_size_sec"] = float(window_seconds)
        rows.append(row)

    return rows


def extract_multiscale_feature_rows(
    packets: Sequence[PacketRecord],
    experiment_id: str,
    burst_gap_sec: float = 0.05,
    idle_threshold_sec: float = 0.5,
    window_sizes_sec: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
    packet_information_threshold: int = 2,
) -> List[Dict[str, Any]]:
    """Return one complete-trace row and online-compatible windows.

    Every window uses only packets observed inside that interval. Multiple
    scales are kept in one long-form feature table and distinguished by
    ``window_size_sec``.
    """
    normalized: List[float] = []
    for value in window_sizes_sec:
        size = float(value)
        if size <= 0:
            raise ValueError("window_sizes_sec values must be positive")
        if size not in normalized:
            normalized.append(size)
    normalized.sort()

    overall = extract_feature_rows(
        packets=packets,
        experiment_id=experiment_id,
        burst_gap_sec=burst_gap_sec,
        idle_threshold_sec=idle_threshold_sec,
        window_seconds=None,
        packet_information_threshold=packet_information_threshold,
    )[0]
    overall["window_size_sec"] = 0.0
    rows: List[Dict[str, Any]] = [overall]

    for size in normalized:
        scale_rows = extract_feature_rows(
            packets=packets,
            experiment_id=experiment_id,
            burst_gap_sec=burst_gap_sec,
            idle_threshold_sec=idle_threshold_sec,
            window_seconds=size,
            packet_information_threshold=packet_information_threshold,
        )
        rows.extend(scale_rows[1:])
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



def write_fingerprint_sequence_csv(
    packets: Sequence[PacketRecord],
    experiment_id: str,
    output_path: str | Path,
) -> Path:
    """Write classifier-safe packet sequence without endpoint identity."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first_timestamp = packets[0].timestamp_epoch if packets else 0.0

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(SAFE_SEQUENCE_FIELDS),
        )
        writer.writeheader()
        for packet in packets:
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "packet_index": packet.index,
                    "relative_time_sec": max(
                        0.0,
                        packet.timestamp_epoch - first_timestamp,
                    ),
                    "frame_length": packet.frame_length,
                    "direction": packet.direction,
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


def _safe_alias(value: str, fallback: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value).strip()
    ).strip("_")
    return cleaned or fallback


def _resolved_client_aliases(
    client_ips: Sequence[str],
    client_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    supplied = client_aliases or {}
    resolved: Dict[str, str] = {}
    used: set[str] = set()
    for index, ip in enumerate(client_ips, start=1):
        fallback = f"trace_{index:03d}"
        alias = _safe_alias(
            str(supplied.get(ip, fallback)),
            fallback,
        )
        candidate = alias
        suffix = 2
        while candidate in used:
            candidate = f"{alias}_{suffix}"
            suffix += 1
        used.add(candidate)
        resolved[ip] = candidate
    return resolved


def _feature_count(row: Dict[str, Any]) -> int:
    metadata = {
        "experiment_id",
        "client_capture_id",
        "row_type",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "window_size_sec",
        "trace_start_offset_sec",
        "trace_end_offset_sec",
        "window_start_global_sec",
        "window_end_global_sec",
        "packet_information_threshold",
        "packet_information_ok",
    }
    return len([key for key in row if key not in metadata])


def write_manifest(
    experiment_id: str,
    source_capture: Dict[str, Any],
    packet_csv_path: Path,
    fingerprint_sequence_csv_path: Path,
    feature_csv_path: Path,
    manifest_path: Path,
    packets: Sequence[PacketRecord],
    feature_rows: Sequence[Dict[str, Any]],
    server_ip: Optional[str],
    client_ip: Optional[str],
    client_ips: Sequence[str],
    client_aliases: Dict[str, str],
    per_client_outputs: Dict[str, Dict[str, Any]],
    capture_interface: Optional[str],
    capture_preflight: Optional[Dict[str, Any]],
    burst_gap_sec: float,
    idle_threshold_sec: float,
    window_seconds: Optional[float],
    window_sizes_sec: Optional[Sequence[float]] = None,
    capture_isolation_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    mtu = None
    if capture_preflight:
        mtu = capture_preflight.get("mtu")
    quality = capture_quality_diagnostics(
        packets,
        interface_mtu=mtu,
    )

    multi_client = len(client_ips) > 1
    manifest = {
        "experiment_id": experiment_id,
        "generated_at_utc": _utc_now_iso(),
        "schemas": {
            "packet_sequence_raw": PACKET_SCHEMA_VERSION,
            "packet_sequence_fingerprint_safe": "1.1",
            "handcrafted_features": FEATURE_SCHEMA_VERSION,
        },
        "capture": source_capture,
        "capture_isolation": {
            "mode": (
                (capture_isolation_metadata or {}).get("mode")
                or (
                    "client_ip_bpf_and_postfilter"
                    if client_ips
                    else "legacy_unisolated"
                )
            ),
            "client_facing_only": bool(client_ips),
            "configured_client_ips": list(client_ips),
            "client_aliases": client_aliases,
            "upstream_duplicate_leg_excluded": bool(client_ips),
            "note": (
                "Client IPs are used only to isolate/group observable "
                "network traffic. Endpoint identity is removed from "
                "classifier-safe sequences."
            ),
            **dict(capture_isolation_metadata or {}),
        },
        "direction_reference": {
            "server_ip": server_ip,
            "client_ip": client_ip,
            "client_ips": list(client_ips),
            "meaning": {
                "up": "client to proxy/server",
                "down": "proxy/server to client",
                "unknown": "direction could not be resolved",
            },
        },
        "feature_parameters": {
            "burst_gap_sec": burst_gap_sec,
            "idle_threshold_sec": idle_threshold_sec,
            "window_seconds": window_seconds,
            "window_sizes_sec": (
                [float(value) for value in window_sizes_sec]
                if window_sizes_sec
                else None
            ),
            "overall_row_included": True,
        },
        "capture_interface": {
            "name": capture_interface,
            "preflight": capture_preflight or {},
        },
        "capture_quality": quality,
        "outputs": {
            "packet_sequence_raw_csv": str(packet_csv_path),
            "packet_sequence_fingerprint_csv": str(
                fingerprint_sequence_csv_path
            ),
            "features_csv": str(feature_csv_path),
            "feature_row_count": len(feature_rows),
            "per_client": per_client_outputs,
        },
        "extractor": {
            "parser": source_capture.get("parser", "tshark"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "fingerprinting_policy": {
            "predictor_source": "proxy_only",
            "contains_ai_ground_truth_labels": False,
            "contains_client_server_resource_telemetry": False,
            "raw_packet_sequence_classifier_eligible": False,
            "fingerprint_sequence_classifier_eligible": not multi_client,
            "per_client_fingerprint_sequences_classifier_eligible": True,
            "combined_multi_client_trace_classifier_eligible": not multi_client,
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
                "excluded. For multi-client captures, use per-client "
                "classifier-safe sequences/features rather than the mixed "
                "combined trace."
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _export_packets_artifacts(
    packets: Sequence[PacketRecord],
    experiment_id: str,
    target_dir: Path,
    source_capture: Dict[str, Any],
    server_ip: Optional[str] = None,
    client_ip: Optional[str] = None,
    client_ips: Optional[Sequence[str]] = None,
    client_aliases: Optional[Dict[str, str]] = None,
    client_connections: Optional[Sequence[Dict[str, Any]]] = None,
    proxy_ip: Optional[str] = None,
    proxy_port: Optional[int] = None,
    per_client_artifacts: bool = True,
    capture_interface: Optional[str] = None,
    capture_preflight: Optional[Dict[str, Any]] = None,
    burst_gap_sec: float = 0.05,
    idle_threshold_sec: float = 0.5,
    window_seconds: Optional[float] = 5.0,
    window_sizes_sec: Optional[Sequence[float]] = None,
    filename_suffix: str = "",
    capture_isolation_metadata: Optional[Dict[str, Any]] = None,
    packet_information_threshold: int = 2,
) -> Dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    clients = _normalized_client_ips(
        client_ip=client_ip,
        client_ips=client_ips,
    )
    aliases = _resolved_client_aliases(
        clients,
        client_aliases=client_aliases,
    )

    stem = f"{experiment_id}{filename_suffix}"
    packet_csv = target_dir / f"{stem}_packet_sequence.csv"
    fingerprint_sequence_csv = (
        target_dir / f"{stem}_fingerprint_sequence.csv"
    )
    feature_csv = target_dir / f"{stem}_features.csv"
    manifest_json = target_dir / f"{stem}_manifest.json"

    feature_rows = (
        extract_multiscale_feature_rows(
            packets=packets,
            experiment_id=experiment_id,
            burst_gap_sec=burst_gap_sec,
            idle_threshold_sec=idle_threshold_sec,
            window_sizes_sec=window_sizes_sec,
            packet_information_threshold=packet_information_threshold,
        )
        if window_sizes_sec
        else extract_feature_rows(
            packets=packets,
            experiment_id=experiment_id,
            burst_gap_sec=burst_gap_sec,
            idle_threshold_sec=idle_threshold_sec,
            window_seconds=window_seconds,
            packet_information_threshold=packet_information_threshold,
        )
    )

    write_packet_sequence_csv(
        packets=packets,
        experiment_id=experiment_id,
        output_path=packet_csv,
    )
    write_fingerprint_sequence_csv(
        packets=packets,
        experiment_id=experiment_id,
        output_path=fingerprint_sequence_csv,
    )
    write_feature_csv(
        rows=feature_rows,
        output_path=feature_csv,
    )

    per_client_outputs: Dict[str, Dict[str, Any]] = {}
    normalized_connections = []
    for item in list(client_connections or []):
        try:
            ip = str(item.get("client_ip", "")).strip()
            port = int(item.get("client_port"))
        except (TypeError, ValueError):
            continue
        if not ip or port <= 0:
            continue
        key = _connection_key(ip, port)
        normalized_connections.append((ip, port, key, str(item.get("alias", "")).strip()))

    if per_client_artifacts and normalized_connections:
        for ip, client_port, connection_key, supplied_alias in normalized_connections:
            client_packets = connection_facing_packets(
                packets,
                client_ip=ip,
                client_port=client_port,
                proxy_ip=proxy_ip,
                proxy_port=proxy_port,
            )
            if not client_packets:
                continue
            alias = supplied_alias or (client_aliases or {}).get(connection_key) or f"trace_{len(per_client_outputs)+1:03d}"
            client_stem = f"{stem}__{alias}"
            client_safe = (
                target_dir
                / f"{client_stem}_fingerprint_sequence.csv"
            )
            client_features = target_dir / f"{client_stem}_features.csv"

            write_fingerprint_sequence_csv(
                packets=client_packets,
                experiment_id=experiment_id,
                output_path=client_safe,
            )
            rows = (
                extract_multiscale_feature_rows(
                    packets=client_packets,
                    experiment_id=experiment_id,
                    burst_gap_sec=burst_gap_sec,
                    idle_threshold_sec=idle_threshold_sec,
                    window_sizes_sec=window_sizes_sec,
                )
                if window_sizes_sec
                else extract_feature_rows(
                    packets=client_packets,
                    experiment_id=experiment_id,
                    burst_gap_sec=burst_gap_sec,
                    idle_threshold_sec=idle_threshold_sec,
                    window_seconds=window_seconds,
                )
            )
            combined_start_epoch = (
                packets[0].timestamp_epoch if packets else 0.0
            )
            trace_start_offset_sec = max(
                0.0,
                client_packets[0].timestamp_epoch
                - combined_start_epoch,
            )
            trace_end_offset_sec = max(
                trace_start_offset_sec,
                client_packets[-1].timestamp_epoch
                - combined_start_epoch,
            )

            rows_with_client: List[Dict[str, Any]] = []
            for row in rows:
                window_start_local = float(
                    row.get("window_start_sec", 0.0)
                )
                window_end_local = float(
                    row.get("window_end_sec", 0.0)
                )
                enriched = {
                    "experiment_id": row["experiment_id"],
                    "client_capture_id": alias,
                    "trace_start_offset_sec": (
                        trace_start_offset_sec
                    ),
                    "trace_end_offset_sec": (
                        trace_end_offset_sec
                    ),
                    "window_start_global_sec": (
                        trace_start_offset_sec
                        + window_start_local
                    ),
                    "window_end_global_sec": (
                        trace_start_offset_sec
                        + window_end_local
                    ),
                }
                enriched.update(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "experiment_id"
                    }
                )
                rows_with_client.append(enriched)

            write_feature_csv(
                rows=rows_with_client,
                output_path=client_features,
            )
            per_client_outputs[alias] = {
                "client_ip": ip,
                "client_port": client_port,
                "connection_key": connection_key,
                "packet_count": len(client_packets),
                "fingerprint_sequence_csv": str(client_safe),
                "features_csv": str(client_features),
                "feature_row_count": len(rows_with_client),
                "trace_start_offset_sec": trace_start_offset_sec,
                "trace_end_offset_sec": trace_end_offset_sec,
                "global_time_reference": (
                    "combined_capture_start"
                ),
                "classifier_eligible": True,
            }
    elif per_client_artifacts and clients:
        for ip in clients:
            client_packets = client_facing_packets(packets, client_ip=ip)
            if not client_packets:
                continue
            alias = aliases[ip]
            client_stem = f"{stem}__{alias}"
            client_safe = target_dir / f"{client_stem}_fingerprint_sequence.csv"
            client_features = target_dir / f"{client_stem}_features.csv"
            write_fingerprint_sequence_csv(packets=client_packets, experiment_id=experiment_id, output_path=client_safe)
            rows = extract_multiscale_feature_rows(
                packets=client_packets, experiment_id=experiment_id, burst_gap_sec=burst_gap_sec,
                idle_threshold_sec=idle_threshold_sec, window_sizes_sec=window_sizes_sec,
                packet_information_threshold=packet_information_threshold
            ) if window_sizes_sec else extract_feature_rows(
                packets=client_packets, experiment_id=experiment_id, burst_gap_sec=burst_gap_sec,
                idle_threshold_sec=idle_threshold_sec, window_seconds=window_seconds,
                packet_information_threshold=packet_information_threshold
            )
            combined_start_epoch = packets[0].timestamp_epoch if packets else 0.0
            trace_start_offset_sec = max(0.0, client_packets[0].timestamp_epoch-combined_start_epoch)
            trace_end_offset_sec = max(trace_start_offset_sec, client_packets[-1].timestamp_epoch-combined_start_epoch)
            rows_with_client=[]
            for row in rows:
                enriched={
                    "experiment_id": row["experiment_id"], "client_capture_id": alias,
                    "trace_start_offset_sec": trace_start_offset_sec, "trace_end_offset_sec": trace_end_offset_sec,
                    "window_start_global_sec": trace_start_offset_sec+float(row.get("window_start_sec",0.0)),
                    "window_end_global_sec": trace_start_offset_sec+float(row.get("window_end_sec",0.0)),
                }
                enriched.update({k:v for k,v in row.items() if k != "experiment_id"})
                rows_with_client.append(enriched)
            write_feature_csv(rows=rows_with_client, output_path=client_features)
            per_client_outputs[alias]={
                "client_ip": ip, "packet_count": len(client_packets),
                "fingerprint_sequence_csv": str(client_safe), "features_csv": str(client_features),
                "feature_row_count": len(rows_with_client), "trace_start_offset_sec": trace_start_offset_sec,
                "trace_end_offset_sec": trace_end_offset_sec, "global_time_reference":"combined_capture_start",
                "classifier_eligible": True,
            }

    isolation_metadata = dict(capture_isolation_metadata or {})
    if normalized_connections:
        isolation_metadata["granularity"] = "accepted_tcp_connection"
        isolation_metadata["accepted_connections"] = [
            {"client_ip": ip, "client_port": port, "alias": (alias or (client_aliases or {}).get(key))}
            for ip, port, key, alias in normalized_connections
        ]

    write_manifest(
        experiment_id=experiment_id,
        source_capture=source_capture,
        packet_csv_path=packet_csv,
        fingerprint_sequence_csv_path=fingerprint_sequence_csv,
        feature_csv_path=feature_csv,
        manifest_path=manifest_json,
        packets=packets,
        feature_rows=feature_rows,
        server_ip=server_ip,
        client_ip=client_ip,
        client_ips=clients,
        client_aliases=aliases,
        per_client_outputs=per_client_outputs,
        capture_interface=capture_interface,
        capture_preflight=capture_preflight,
        burst_gap_sec=burst_gap_sec,
        idle_threshold_sec=idle_threshold_sec,
        window_seconds=window_seconds,
        window_sizes_sec=window_sizes_sec,
        capture_isolation_metadata=isolation_metadata,
    )

    quality = capture_quality_diagnostics(
        packets,
        interface_mtu=(
            (capture_preflight or {}).get("mtu")
        ),
    )
    if quality["possible_offload_coalescing"]:
        print(
            "[traffic] WARNING: observed "
            f"{quality['oversized_frame_count']} frames above "
            f"{quality['oversized_frame_threshold_bytes']} bytes "
            f"(max={quality['max_frame_length_bytes']}). "
            "Inspect/disable GRO/GSO/TSO/LRO on the capture interface "
            "before final packet-size experiments."
        )

    overall = feature_rows[0]
    return {
        "experiment_id": experiment_id,
        "packet_sequence_csv": str(packet_csv),
        "fingerprint_sequence_csv": str(fingerprint_sequence_csv),
        "features_csv": str(feature_csv),
        "manifest_json": str(manifest_json),
        "packet_count": len(packets),
        "feature_count": _feature_count(overall),
        "feature_row_count": len(feature_rows),
        "per_client_artifacts": per_client_outputs,
        "capture_quality": quality,
    }


def extract_capture_artifacts(
    pcap_path: str | Path,
    experiment_id: str,
    output_dir: str | Path | None = None,
    server_ip: Optional[str] = None,
    client_ip: Optional[str] = None,
    client_ips: Optional[Sequence[str]] = None,
    client_aliases: Optional[Dict[str, str]] = None,
    client_connections: Optional[Sequence[Dict[str, Any]]] = None,
    proxy_ip: Optional[str] = None,
    proxy_port: Optional[int] = None,
    per_client_artifacts: bool = True,
    capture_interface: Optional[str] = None,
    capture_preflight: Optional[Dict[str, Any]] = None,
    burst_gap_sec: float = 0.05,
    idle_threshold_sec: float = 0.5,
    window_seconds: Optional[float] = 5.0,
    window_sizes_sec: Optional[Sequence[float]] = None,
    capture_isolation_metadata: Optional[Dict[str, Any]] = None,
    packet_information_threshold: int = 2,
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

    clients = _normalized_client_ips(
        client_ip=client_ip,
        client_ips=client_ips,
    )
    packets = read_pcap_with_tshark(
        pcap_path=pcap,
        server_ip=server_ip,
        client_ip=client_ip,
        client_ips=clients,
        isolate_client_facing=bool(clients),
    )

    source_capture = {
        "source_kind": "pcap",
        "parser": "tshark",
        "pcap_path": str(pcap),
        "pcap_size_bytes": pcap.stat().st_size,
        "pcap_sha256": _sha256(pcap),
        "packet_count": len(packets),
        "capture_start_epoch": (
            packets[0].timestamp_epoch if packets else None
        ),
        "capture_end_epoch": (
            packets[-1].timestamp_epoch if packets else None
        ),
    }

    result = _export_packets_artifacts(
        packets=packets,
        experiment_id=experiment_id,
        target_dir=target_dir,
        source_capture=source_capture,
        server_ip=server_ip,
        client_ip=client_ip,
        client_ips=clients,
        client_aliases=client_aliases,
        client_connections=client_connections,
        proxy_ip=proxy_ip,
        proxy_port=proxy_port,
        per_client_artifacts=per_client_artifacts,
        capture_interface=capture_interface,
        capture_preflight=capture_preflight,
        burst_gap_sec=burst_gap_sec,
        idle_threshold_sec=idle_threshold_sec,
        window_seconds=window_seconds,
        window_sizes_sec=window_sizes_sec,
        capture_isolation_metadata=capture_isolation_metadata,
        packet_information_threshold=packet_information_threshold,
    )
    result["pcap"] = str(pcap)
    return result


def repair_packet_sequence_artifacts(
    raw_packet_csv: str | Path,
    experiment_id: str,
    client_ips: Sequence[str],
    output_dir: str | Path | None = None,
    client_aliases: Optional[Dict[str, str]] = None,
    server_ip: Optional[str] = None,
    burst_gap_sec: float = 0.05,
    idle_threshold_sec: float = 0.5,
    window_seconds: Optional[float] = 5.0,
) -> Dict[str, Any]:
    """Repair an existing broad proxy packet CSV without the original PCAP."""
    source = Path(raw_packet_csv)
    if not source.exists():
        raise FeatureExtractionError(
            f"Raw packet sequence not found: {source}"
        )
    clients = _normalized_client_ips(client_ips=client_ips)
    if not clients:
        raise FeatureExtractionError(
            "At least one client IP is required to repair/isolate a proxy "
            "packet sequence"
        )

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else source.parent / f"{experiment_id}_repaired"
    )
    packets = read_packet_sequence_csv(
        csv_path=source,
        client_ips=clients,
        server_ip=server_ip,
        isolate_client_facing=True,
    )
    source_capture = {
        "source_kind": "raw_packet_sequence_repair",
        "parser": "csv",
        "source_packet_sequence_csv": str(source),
        "source_packet_sequence_size_bytes": source.stat().st_size,
        "source_packet_sequence_sha256": _sha256(source),
        "packet_count_after_client_facing_filter": len(packets),
        "capture_start_epoch": (
            packets[0].timestamp_epoch if packets else None
        ),
        "capture_end_epoch": (
            packets[-1].timestamp_epoch if packets else None
        ),
    }
    result = _export_packets_artifacts(
        packets=packets,
        experiment_id=experiment_id,
        target_dir=target_dir,
        source_capture=source_capture,
        server_ip=server_ip,
        client_ips=clients,
        client_aliases=client_aliases,
        per_client_artifacts=True,
        burst_gap_sec=burst_gap_sec,
        idle_threshold_sec=idle_threshold_sec,
        window_seconds=window_seconds,
        filename_suffix="_repaired",
    )
    result["source_packet_sequence_csv"] = str(source)
    result["repair_output_dir"] = str(target_dir)
    return result

