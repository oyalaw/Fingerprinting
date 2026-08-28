from __future__ import annotations

import copy

from ai_fingerprint.live_inference import LiveArchitectureMonitor
from ai_fingerprint.traffic.analysis import PacketRecord
from ai_fingerprint.proxy import BlindTCPProxy, DEFAULT_PROXY_CONFIG


def test_proxy_registers_clients_with_neutral_stable_aliases(tmp_path):
    config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
    config["experiment"]["experiment_id"] = "AUTO_DISCOVERY"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["capture"]["enabled"] = False
    proxy = BlindTCPProxy(config)

    first = proxy._register_discovered_client(
        "10.42.0.47", source="test", notify_monitor=False
    )
    second = proxy._register_discovered_client(
        "10.42.0.210", source="test", notify_monitor=False
    )
    duplicate = proxy._register_discovered_client(
        "10.42.0.47", source="test", notify_monitor=False
    )

    assert first == "trace_001"
    assert second == "trace_002"
    assert duplicate == "trace_001"
    assert proxy._capture_client_ips() == ["10.42.0.47", "10.42.0.210"]
    assert proxy._capture_client_aliases() == {
        "10.42.0.47": "trace_001",
        "10.42.0.210": "trace_002",
    }


def test_proxy_never_registers_upstream_or_its_own_listen_ip(tmp_path):
    config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
    config["experiment"]["experiment_id"] = "AUTO_DISCOVERY"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["capture"]["enabled"] = False
    proxy = BlindTCPProxy(config)

    assert proxy._register_discovered_client(
        "10.42.0.195", source="test", notify_monitor=False
    ) == ""
    assert proxy._register_discovered_client(
        "10.42.0.1", source="test", notify_monitor=False
    ) == ""
    assert proxy._capture_client_ips() == []


def test_live_monitor_can_register_auto_discovered_client(tmp_path):
    monitor = LiveArchitectureMonitor(
        experiment_id="LIVE_DISCOVERY",
        interface="wlan0",
        client_ips=[],
        client_aliases={},
        port=8080,
        proxy_ip="10.42.0.1",
        exclude_hosts=["10.42.0.195"],
        output_dir=tmp_path,
        window_sizes_sec=[0.5, 1.0, 2.0, 5.0],
        burst_gap_sec=0.05,
        idle_threshold_sec=0.5,
        model_root=tmp_path / "models",
    )
    alias = monitor.register_client("10.42.0.47")
    assert alias == "trace_001"
    assert monitor.client_ips == ["10.42.0.47"]
    assert monitor.aliases["10.42.0.47"] == "trace_001"


def test_live_monitor_flushes_pre_accept_handshake_packet(tmp_path):
    monitor = LiveArchitectureMonitor(
        experiment_id="LIVE_PENDING",
        interface="wlan0",
        client_ips=[],
        client_aliases={},
        port=8080,
        proxy_ip="10.42.0.1",
        exclude_hosts=["10.42.0.195"],
        output_dir=tmp_path,
        window_sizes_sec=[0.5],
        burst_gap_sec=0.05,
        idle_threshold_sec=0.5,
        model_root=tmp_path / "models",
    )
    packet = PacketRecord(
        index=1,
        timestamp_epoch=100.0,
        frame_length=74,
        src_ip="10.42.0.47",
        dst_ip="10.42.0.1",
        src_port=50000,
        dst_port=8080,
        transport_protocol="tcp",
        tcp_flags_hex="0x0002",
        tcp_syn=1,
        tcp_ack=0,
        tcp_fin=0,
        tcp_rst=0,
        retransmission=0,
        tls_record_lengths=(),
        direction="unknown",
    )
    monitor._pending_packets["10.42.0.47"] = __import__(
        "collections"
    ).deque([packet], maxlen=256)
    monitor.register_client("10.42.0.47", alias="trace_001")
    state = monitor._traces["trace_001"]
    assert len(state.packets) == 1
    assert state.packets[0].direction == "up"
