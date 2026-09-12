from __future__ import annotations

from ai_fingerprint.round_fingerprinting import (
    RoundInferenceConfig,
    infer_round_boundaries,
)
from ai_fingerprint.traffic.analysis import PacketRecord


def packet(index, t, direction, size=1500):
    return PacketRecord(
        index=index,
        timestamp_epoch=t,
        frame_length=size,
        src_ip="10.0.0.2" if direction == "up" else "10.0.0.1",
        dst_ip="10.0.0.1" if direction == "up" else "10.0.0.2",
        src_port=50000 if direction == "up" else 8080,
        dst_port=8080 if direction == "up" else 50000,
        transport_protocol="TCP",
        tcp_flags_hex="0x0010",
        tcp_syn=0,
        tcp_ack=1,
        tcp_fin=0,
        tcp_rst=0,
        retransmission=0,
        tls_record_lengths=tuple(),
        direction=direction,
    )


def test_round_detector_pairs_download_training_upload_cycles():
    packets = []
    idx = 1
    base = 1000.0
    for round_index in range(5):
        start = base + round_index * 10.0
        # Large down transfer.
        for k in range(100):
            packets.append(packet(idx, start + k * 0.005, "down")); idx += 1
        # Training gap: no packets for ~3s.
        up_start = start + 4.0
        for k in range(100):
            packets.append(packet(idx, up_start + k * 0.005, "up")); idx += 1

    cfg = RoundInferenceConfig(
        bin_sec=0.1,
        min_bin_bytes=1500,
        bridge_gap_sec=0.2,
        min_major_transfer_bytes=50_000,
        min_round_packets=20,
    )
    rounds, diag = infer_round_boundaries(packets, cfg)
    assert diag["status"] == "ok"
    assert len(rounds) == 5
    assert all(r.download_bytes > 100_000 for r in rounds)
    assert all(r.upload_bytes > 100_000 for r in rounds)
    assert all(r.training_gap_sec > 2.0 for r in rounds)


def test_round_index_is_not_needed_by_detector():
    packets = []
    idx = 1
    for k in range(80):
        packets.append(packet(idx, 1.0 + k * 0.003, "down")); idx += 1
    for k in range(80):
        packets.append(packet(idx, 3.0 + k * 0.003, "up")); idx += 1
    rounds, _ = infer_round_boundaries(
        packets,
        RoundInferenceConfig(
            bin_sec=0.1,
            min_bin_bytes=1500,
            bridge_gap_sec=0.2,
            min_major_transfer_bytes=40_000,
            min_round_packets=20,
        ),
    )
    assert len(rounds) == 1
    assert rounds[0].inferred_round_index == 0
