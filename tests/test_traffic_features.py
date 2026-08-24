from __future__ import annotations

import csv
import json

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
