from __future__ import annotations

import pytest

import ai_fingerprint.offload as offload
from ai_fingerprint.offload import (
    CaptureOffloadError,
    CaptureOffloadManager,
    parse_ethtool_features,
)
from ai_fingerprint.proxy import DEFAULT_PROXY_CONFIG


def _state(gro: bool, *, gso=False, tso=False, lro=False):
    values = {
        "gro": gro,
        "gso": gso,
        "tso": tso,
        "lro": lro,
    }
    return {
        "interface": "wlan0",
        "available": True,
        "read_ok": True,
        "features": {
            name: {
                "kernel_name": offload.OFFLOAD_FEATURES[name],
                "supported": True,
                "enabled": enabled,
                "state": "on" if enabled else "off",
                "fixed": False,
            }
            for name, enabled in values.items()
        },
    }


def test_parse_ethtool_features_includes_fixed_marker():
    parsed = parse_ethtool_features(
        """
Features for wlan0:
tcp-segmentation-offload: on
generic-segmentation-offload: off
generic-receive-offload: on
large-receive-offload: off [fixed]
"""
    )
    assert parsed["generic-receive-offload"]["enabled"] is True
    assert parsed["large-receive-offload"]["enabled"] is False
    assert parsed["large-receive-offload"]["fixed"] is True


def test_manager_disables_and_restores_only_changed_features(
    monkeypatch,
    tmp_path,
):
    states = iter([
        _state(True),
        _state(False),
        _state(True),
    ])
    monkeypatch.setattr(
        offload,
        "read_offload_state",
        lambda interface: next(states),
    )

    actions = []

    def fake_set(**kwargs):
        actions.append(
            (kwargs["feature"], kwargs["state"])
        )
        return {
            "ok": True,
            "feature": kwargs["feature"],
            "requested_state": kwargs["state"],
            "method": "test",
            "attempts": [],
        }

    monkeypatch.setattr(
        offload,
        "_run_set_command",
        fake_set,
    )

    manager = CaptureOffloadManager(
        "wlan0",
        state_path=tmp_path / "offload.json",
    )
    manager.start()

    assert manager.status == "capture_safe"
    assert manager.changed_features == ["gro"]
    assert actions == [("gro", "off")]

    manager.restore()
    assert manager.status == "restored"
    assert actions == [
        ("gro", "off"),
        ("gro", "on"),
    ]
    assert (tmp_path / "offload.json").exists()


def test_required_manager_fails_closed_and_restores_partial_change(
    monkeypatch,
):
    # The disable command reports success, but verification still observes
    # GRO enabled. The manager must restore what it changed and abort.
    states = iter([
        _state(True),
        _state(True),
        _state(True),
    ])
    monkeypatch.setattr(
        offload,
        "read_offload_state",
        lambda interface: next(states),
    )

    actions = []

    def fake_set(**kwargs):
        actions.append(
            (kwargs["feature"], kwargs["state"])
        )
        return {
            "ok": True,
            "feature": kwargs["feature"],
            "requested_state": kwargs["state"],
            "method": "test",
            "attempts": [],
        }

    monkeypatch.setattr(
        offload,
        "_run_set_command",
        fake_set,
    )

    manager = CaptureOffloadManager(
        "wlan0",
        required=True,
    )
    with pytest.raises(
        CaptureOffloadError,
        match="Capture aborted",
    ):
        manager.start()

    assert actions == [
        ("gro", "off"),
        ("gro", "on"),
    ]
    assert manager.restored is True


def test_proxy_defaults_require_offload_fidelity_protection():
    config = DEFAULT_PROXY_CONFIG["capture"]["offload_management"]
    assert config["enabled"] is True
    assert config["required"] is True
    assert config["restore_on_exit"] is True
    assert config["features"] == ["gro", "gso", "tso", "lro"]
