from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_fingerprint.traffic.analysis import (
    PacketRecord,
    extract_feature_rows,
    write_feature_csv,
    write_packet_sequence_csv,
)


def packet(
    index: int,
    timestamp: float,
    size: int,
    direction: str,
    src: str,
    dst: str,
    src_port: int = 50000,
    dst_port: int = 5000,
    syn: int = 0,
    ack: int = 1,
    fin: int = 0,
    rst: int = 0,
    retransmission: int = 0,
    tls_lengths=(),
):
    return PacketRecord(
        index=index,
        timestamp_epoch=timestamp,
        frame_length=size,
        src_ip=src,
        dst_ip=dst,
        src_port=src_port,
        dst_port=dst_port,
        transport_protocol="TCP",
        tcp_flags_hex="0x0010",
        tcp_syn=syn,
        tcp_ack=ack,
        tcp_fin=fin,
        tcp_rst=rst,
        retransmission=retransmission,
        tls_record_lengths=tuple(tls_lengths),
        direction=direction,
    )


def sample_packets():
    return [
        packet(
            1,
            1000.00,
            100,
            "up",
            "10.0.0.2",
            "10.0.0.1",
            syn=1,
            ack=0,
        ),
        packet(
            2,
            1000.01,
            200,
            "up",
            "10.0.0.2",
            "10.0.0.1",
            tls_lengths=(180,),
        ),
        packet(
            3,
            1000.03,
            300,
            "down",
            "10.0.0.1",
            "10.0.0.2",
            src_port=5000,
            dst_port=50000,
            tls_lengths=(250,),
        ),
        packet(
            4,
            1000.80,
            400,
            "down",
            "10.0.0.1",
            "10.0.0.2",
            src_port=5000,
            dst_port=50000,
            retransmission=1,
        ),
    ]


def test_overall_feature_extraction():
    rows = extract_feature_rows(
        sample_packets(),
        experiment_id="EXP_TEST",
        burst_gap_sec=0.05,
        idle_threshold_sec=0.5,
    )

    assert len(rows) == 1
    row = rows[0]

    assert row["packet_count_total"] == 4
    assert row["packet_count_up"] == 2
    assert row["packet_count_down"] == 2
    assert row["bytes_total"] == 1000
    assert row["bytes_up"] == 300
    assert row["bytes_down"] == 700
    assert row["burst_count_total"] == 3
    assert row["idle_gap_count"] == 1
    assert row["tcp_syn_count"] == 1
    assert row["tcp_retransmission_count"] == 1
    assert row["tls_record_count"] == 2
    assert row["connection_count"] == 1


def test_window_feature_extraction():
    rows = extract_feature_rows(
        sample_packets(),
        experiment_id="EXP_TEST",
        window_seconds=0.5,
    )

    assert rows[0]["row_type"] == "overall"
    assert any(row["row_type"] == "window" for row in rows)


def test_packet_and_feature_csv_export(tmp_path):
    packets = sample_packets()
    rows = extract_feature_rows(
        packets,
        experiment_id="EXP_TEST",
    )

    packet_csv = tmp_path / "packets.csv"
    feature_csv = tmp_path / "features.csv"

    write_packet_sequence_csv(
        packets,
        experiment_id="EXP_TEST",
        output_path=packet_csv,
    )
    write_feature_csv(rows, feature_csv)

    with packet_csv.open(newline="", encoding="utf-8") as handle:
        packet_rows = list(csv.DictReader(handle))

    with feature_csv.open(newline="", encoding="utf-8") as handle:
        feature_rows = list(csv.DictReader(handle))

    assert len(packet_rows) == 4
    assert packet_rows[0]["direction"] == "up"
    assert packet_rows[1]["tls_record_lengths"] == "180"
    assert len(feature_rows) == 1
    assert feature_rows[0]["experiment_id"] == "EXP_TEST"


def test_feature_vector_is_large_enough_for_hybrid_baseline():
    rows = extract_feature_rows(
        sample_packets(),
        experiment_id="EXP_TEST",
    )
    metadata_fields = {
        "experiment_id",
        "row_type",
        "window_index",
        "window_start_sec",
        "window_end_sec",
    }
    feature_names = set(rows[0]) - metadata_fields
    assert len(feature_names) >= 80


def test_raw_sequence_repair_filters_upstream_and_reconstructs_tcp_flags(tmp_path):
    from ai_fingerprint.traffic.analysis import read_packet_sequence_csv

    raw = tmp_path / "EXP_packet_sequence.csv"
    fieldnames = [
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
    ]
    rows = [
        {
            "experiment_id": "EXP",
            "packet_index": "1",
            "timestamp_epoch": "1000.0",
            "relative_time_sec": "0",
            "frame_length": "74",
            "direction": "up",
            "src_ip": "10.42.0.47",
            "dst_ip": "10.42.0.1",
            "src_port": "50000",
            "dst_port": "8080",
            "transport_protocol": "TCP",
            "tcp_flags_hex": "0x0002",
            "tcp_syn": "0",
            "tcp_ack": "0",
            "tcp_fin": "0",
            "tcp_rst": "0",
            "retransmission": "0",
            "tls_record_count": "0",
            "tls_record_lengths": "",
        },
        {
            "experiment_id": "EXP",
            "packet_index": "2",
            "timestamp_epoch": "1000.001",
            "relative_time_sec": "0.001",
            "frame_length": "74",
            "direction": "down",  # old contaminated interpretation
            "src_ip": "10.42.0.1",
            "dst_ip": "10.42.0.195",
            "src_port": "51000",
            "dst_port": "8080",
            "transport_protocol": "TCP",
            "tcp_flags_hex": "0x0002",
            "tcp_syn": "0",
            "tcp_ack": "0",
            "tcp_fin": "0",
            "tcp_rst": "0",
            "retransmission": "0",
            "tls_record_count": "0",
            "tls_record_lengths": "",
        },
        {
            "experiment_id": "EXP",
            "packet_index": "3",
            "timestamp_epoch": "1000.002",
            "relative_time_sec": "0.002",
            "frame_length": "74",
            "direction": "down",
            "src_ip": "10.42.0.1",
            "dst_ip": "10.42.0.47",
            "src_port": "8080",
            "dst_port": "50000",
            "transport_protocol": "TCP",
            "tcp_flags_hex": "0x0012",
            "tcp_syn": "0",
            "tcp_ack": "0",
            "tcp_fin": "0",
            "tcp_rst": "0",
            "retransmission": "0",
            "tls_record_count": "0",
            "tls_record_lengths": "",
        },
    ]
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    packets = read_packet_sequence_csv(
        raw,
        client_ips=["10.42.0.47"],
        isolate_client_facing=True,
    )

    assert len(packets) == 2
    assert all("10.42.0.195" not in {p.src_ip, p.dst_ip} for p in packets)
    assert packets[0].direction == "up"
    assert packets[0].tcp_syn == 1
    assert packets[0].tcp_ack == 0
    assert packets[1].direction == "down"
    assert packets[1].tcp_syn == 1
    assert packets[1].tcp_ack == 1


def test_capture_quality_flags_large_coalesced_frames():
    from ai_fingerprint.traffic.analysis import capture_quality_diagnostics

    packets = sample_packets() + [
        packet(
            5,
            1001.0,
            65000,
            "up",
            "10.0.0.2",
            "10.0.0.1",
        )
    ]
    quality = capture_quality_diagnostics(packets, interface_mtu=1500)
    assert quality["possible_offload_coalescing"] is True
    assert quality["oversized_frame_count"] == 1
    assert quality["max_frame_length_bytes"] == 65000


def test_repair_exports_per_client_and_windowed_artifacts(tmp_path):
    from ai_fingerprint.traffic.analysis import repair_packet_sequence_artifacts

    raw = tmp_path / "EXP_packet_sequence.csv"
    fieldnames = [
        "experiment_id", "packet_index", "timestamp_epoch",
        "relative_time_sec", "frame_length", "direction", "src_ip",
        "dst_ip", "src_port", "dst_port", "transport_protocol",
        "tcp_flags_hex", "tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst",
        "retransmission", "tls_record_count", "tls_record_lengths",
    ]
    rows = []
    index = 1
    for client, base_port in [("10.42.0.47", 50000), ("10.42.0.210", 51000)]:
        for timestamp, src, dst, flags in [
            (1000.0, client, "10.42.0.1", "0x0002"),
            (1000.1, "10.42.0.1", client, "0x0012"),
            (1005.1, client, "10.42.0.1", "0x0018"),
        ]:
            rows.append({
                "experiment_id": "EXP",
                "packet_index": str(index),
                "timestamp_epoch": str(timestamp),
                "relative_time_sec": "0",
                "frame_length": "1514",
                "direction": "unknown",
                "src_ip": src,
                "dst_ip": dst,
                "src_port": str(base_port if src == client else 8080),
                "dst_port": str(8080 if dst == "10.42.0.1" else base_port),
                "transport_protocol": "TCP",
                "tcp_flags_hex": flags,
                "tcp_syn": "0",
                "tcp_ack": "0",
                "tcp_fin": "0",
                "tcp_rst": "0",
                "retransmission": "0",
                "tls_record_count": "0",
                "tls_record_lengths": "",
            })
            index += 1
    # One upstream duplicate packet must be removed.
    rows.append({
        "experiment_id": "EXP",
        "packet_index": str(index),
        "timestamp_epoch": "1000.05",
        "relative_time_sec": "0",
        "frame_length": "1514",
        "direction": "down",
        "src_ip": "10.42.0.1",
        "dst_ip": "10.42.0.195",
        "src_port": "52000",
        "dst_port": "8080",
        "transport_protocol": "TCP",
        "tcp_flags_hex": "0x0018",
        "tcp_syn": "0",
        "tcp_ack": "0",
        "tcp_fin": "0",
        "tcp_rst": "0",
        "retransmission": "0",
        "tls_record_count": "0",
        "tls_record_lengths": "",
    })
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = repair_packet_sequence_artifacts(
        raw_packet_csv=raw,
        experiment_id="EXP",
        client_ips=["10.42.0.47", "10.42.0.210"],
        client_aliases={
            "10.42.0.47": "client_1",
            "10.42.0.210": "client_2",
        },
        output_dir=tmp_path / "repaired",
        window_seconds=5.0,
    )

    assert result["packet_count"] == 6
    assert set(result["per_client_artifacts"]) == {"client_1", "client_2"}
    for alias, details in result["per_client_artifacts"].items():
        assert details["packet_count"] == 3
        assert Path(details["features_csv"]).exists()
        assert Path(details["fingerprint_sequence_csv"]).exists()
        with open(details["features_csv"], newline="", encoding="utf-8") as handle:
            feature_rows = list(csv.DictReader(handle))
        assert feature_rows[0]["client_capture_id"] == alias
        assert len(feature_rows) >= 3  # overall + at least two windows
