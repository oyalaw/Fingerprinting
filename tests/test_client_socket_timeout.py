from pathlib import Path


def test_client_connection_timeout_is_connection_only():
    source = Path("ai_fingerprint/client.py").read_text(encoding="utf-8")
    assert "socket.create_connection((host, port), timeout=30)" in source
    assert "sock.settimeout(None)" in source
    assert "tls_sock.settimeout(None)" in source
    assert "SO_KEEPALIVE" in source
