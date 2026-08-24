from __future__ import annotations

import copy
import socket
import threading
import time

from ai_fingerprint.proxy import (
    BlindTCPProxy,
    DEFAULT_PROXY_CONFIG,
    validate_proxy_config,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_proxy_config_is_label_blind():
    config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
    config["capture"]["enabled"] = False
    validate_proxy_config(config)

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
    assert forbidden.isdisjoint(config)


def test_blind_proxy_forwards_bytes(tmp_path):
    upstream_port = _free_port()
    proxy_port = _free_port()

    upstream_ready = threading.Event()
    upstream_stop = threading.Event()

    def echo_server():
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", upstream_port))
        listener.listen(4)
        listener.settimeout(0.2)
        upstream_ready.set()

        try:
            while not upstream_stop.is_set():
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                with conn:
                    while True:
                        data = conn.recv(65536)
                        if not data:
                            break
                        conn.sendall(data)
        finally:
            listener.close()

    upstream_thread = threading.Thread(
        target=echo_server,
        daemon=True,
    )
    upstream_thread.start()
    assert upstream_ready.wait(timeout=2)

    config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
    config["experiment"]["experiment_id"] = "PROXY_TEST"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["proxy"].update(
        {
            "listen_host": "127.0.0.1",
            "listen_port": proxy_port,
            "upstream_host": "127.0.0.1",
            "upstream_port": upstream_port,
        }
    )
    config["capture"]["enabled"] = False
    config["proxy"]["forwarding_log_enabled"] = True

    proxy = BlindTCPProxy(config)
    proxy_thread = threading.Thread(
        target=proxy.serve_forever,
        daemon=True,
    )
    proxy_thread.start()

    deadline = time.time() + 3
    while True:
        try:
            client = socket.create_connection(
                ("127.0.0.1", proxy_port),
                timeout=0.2,
            )
            break
        except OSError:
            if time.time() >= deadline:
                raise
            time.sleep(0.05)

    payload = b"encrypted-looking-test-payload" * 100
    with client:
        client.sendall(payload)
        received = bytearray()
        while len(received) < len(payload):
            chunk = client.recv(65536)
            if not chunk:
                break
            received.extend(chunk)

    assert bytes(received) == payload

    proxy.stop()
    proxy_thread.join(timeout=3)
    upstream_stop.set()
    upstream_thread.join(timeout=2)

    assert (tmp_path / "PROXY_TEST_proxy_forwarding.csv").exists()
    assert (tmp_path / "PROXY_TEST_proxy_summary.json").exists()


def test_capture_requires_client_ips_when_enabled():
    import pytest
    from ai_fingerprint.proxy import ProxyError

    config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
    config["capture"]["interface"] = "wlan0"
    config["capture"]["client_ip"] = None
    config["capture"]["client_ips"] = []
    with pytest.raises(ProxyError, match="client_ips is required"):
        validate_proxy_config(config)


def test_proxy_default_uses_small_pcap_snaplen():
    assert DEFAULT_PROXY_CONFIG["capture"]["snaplen_bytes"] == 256


def test_forwarding_chunk_csv_is_disabled_by_default():
    assert DEFAULT_PROXY_CONFIG["proxy"]["forwarding_log_enabled"] is False
