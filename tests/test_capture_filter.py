from ai_fingerprint.capture import build_capture_filter


def test_multi_client_capture_filter_excludes_unlisted_upstream():
    value = build_capture_filter(
        hosts=["10.42.0.47", "10.42.0.210"],
        port=8080,
    )
    assert value == (
        "(host 10.42.0.47 or host 10.42.0.210) and port 8080"
    )
    assert "10.42.0.195" not in value


def test_automatic_proxy_filter_uses_proxy_host_and_excludes_upstream():
    value = build_capture_filter(
        host="10.42.0.1",
        port=8080,
        exclude_hosts=["10.42.0.195"],
    )
    assert value == (
        "host 10.42.0.1 and port 8080 and not host 10.42.0.195"
    )
