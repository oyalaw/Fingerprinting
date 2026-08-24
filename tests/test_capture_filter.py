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
