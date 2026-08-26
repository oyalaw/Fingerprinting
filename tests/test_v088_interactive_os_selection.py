from __future__ import annotations

from pathlib import Path

from ai_fingerprint import registry


def test_dell_operating_system_choices_are_interactive_and_device_aware():
    choices = registry.operating_systems_for_device("dell_desktop")
    assert choices == [
        "ubuntu",
        "windows_11",
        "windows_10",
        "debian",
        "fedora",
        "other_linux",
        "custom",
    ]


def test_jetson_operating_system_choices_prioritize_ubuntu():
    choices = registry.operating_systems_for_device("jetson_orin_nano")
    assert choices[0] == "ubuntu"
    assert "windows_11" not in choices
    assert choices[-1] == "custom"


def test_mobile_and_apple_devices_have_expected_os_choices():
    assert registry.operating_systems_for_device("android_phone") == [
        "android",
        "custom",
    ]
    assert registry.operating_systems_for_device("iphone") == ["ios", "custom"]
    assert registry.operating_systems_for_device("ipad") == [
        "ipados",
        "ios",
        "custom",
    ]
    assert registry.operating_systems_for_device("macbook")[0] == "macos"


def test_unknown_or_custom_device_gets_full_os_catalogue():
    assert registry.operating_systems_for_device("custom") == registry.OPERATING_SYSTEMS
    assert registry.operating_systems_for_device("future_accelerator") == (
        registry.OPERATING_SYSTEMS
    )


def test_interactive_cli_uses_numbered_os_selection_not_free_text_for_normal_os():
    source = Path("ai_fingerprint/cli.py").read_text(encoding="utf-8")
    assert '"Select operating system"' in source
    assert "registry.operating_systems_for_device" in source
    assert '"Operating system label"' not in source
    # Free text is retained only after explicitly selecting the custom fallback.
    assert '"Custom operating system label"' in source
