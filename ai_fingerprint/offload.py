from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class CaptureOffloadError(RuntimeError):
    pass


OFFLOAD_FEATURES: Dict[str, str] = {
    "gro": "generic-receive-offload",
    "gso": "generic-segmentation-offload",
    "tso": "tcp-segmentation-offload",
    "lro": "large-receive-offload",
}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _ethtool_path() -> Optional[str]:
    return shutil.which("ethtool")


def _sudo_path() -> Optional[str]:
    return shutil.which("sudo")


def parse_ethtool_features(
    text: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse the capture-affecting subset of ``ethtool -k`` output.

    Example lines:
      generic-receive-offload: on
      large-receive-offload: off [fixed]
    """
    wanted = set(OFFLOAD_FEATURES.values())
    parsed: Dict[str, Dict[str, Any]] = {}

    for raw_line in text.splitlines():
        if ":" not in raw_line:
            continue
        name, raw_value = raw_line.split(":", 1)
        name = name.strip()
        if name not in wanted:
            continue

        value = raw_value.strip()
        pieces = value.split()
        if not pieces:
            continue

        state = pieces[0].lower()
        if state not in {"on", "off"}:
            continue

        parsed[name] = {
            "enabled": state == "on",
            "state": state,
            "fixed": "[fixed]" in value.lower(),
        }

    return parsed


def read_offload_state(
    interface: str,
) -> Dict[str, Any]:
    interface = str(interface).strip()
    if not interface:
        raise CaptureOffloadError(
            "Capture interface must not be empty"
        )

    ethtool = _ethtool_path()
    if not ethtool:
        return {
            "timestamp_utc": _utc_now_iso(),
            "interface": interface,
            "ethtool_path": None,
            "available": False,
            "read_ok": False,
            "error": (
                "ethtool is not installed or not available on PATH"
            ),
            "features": {},
        }

    completed = subprocess.run(
        [ethtool, "-k", interface],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if completed.returncode != 0:
        return {
            "timestamp_utc": _utc_now_iso(),
            "interface": interface,
            "ethtool_path": ethtool,
            "available": True,
            "read_ok": False,
            "returncode": completed.returncode,
            "error": (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "ethtool -k failed"
            ),
            "features": {},
        }

    parsed = parse_ethtool_features(completed.stdout)
    features: Dict[str, Dict[str, Any]] = {}
    for short_name, long_name in OFFLOAD_FEATURES.items():
        if long_name not in parsed:
            features[short_name] = {
                "kernel_name": long_name,
                "supported": False,
                "enabled": None,
                "state": "unknown",
                "fixed": None,
            }
            continue

        item = dict(parsed[long_name])
        item.update(
            {
                "kernel_name": long_name,
                "supported": True,
            }
        )
        features[short_name] = item

    return {
        "timestamp_utc": _utc_now_iso(),
        "interface": interface,
        "ethtool_path": ethtool,
        "available": True,
        "read_ok": True,
        "features": features,
    }


def _run_set_command(
    *,
    interface: str,
    feature: str,
    state: str,
    allow_sudo_noninteractive: bool,
) -> Dict[str, Any]:
    ethtool = _ethtool_path()
    if not ethtool:
        return {
            "ok": False,
            "feature": feature,
            "requested_state": state,
            "method": None,
            "error": "ethtool is not installed",
            "attempts": [],
        }

    if feature not in OFFLOAD_FEATURES:
        return {
            "ok": False,
            "feature": feature,
            "requested_state": state,
            "method": None,
            "error": f"Unsupported managed offload feature: {feature}",
            "attempts": [],
        }

    if state not in {"on", "off"}:
        raise ValueError("state must be 'on' or 'off'")

    base = [
        ethtool,
        "-K",
        interface,
        feature,
        state,
    ]
    attempts = []

    def execute(
        command: list[str],
        method: str,
    ) -> Dict[str, Any]:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        attempt = {
            "method": method,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        attempts.append(attempt)
        return attempt

    direct = execute(base, "direct")
    if direct["returncode"] == 0:
        return {
            "ok": True,
            "feature": feature,
            "requested_state": state,
            "method": "direct",
            "attempts": attempts,
        }

    if allow_sudo_noninteractive:
        sudo = _sudo_path()
        if sudo:
            privileged = execute(
                [sudo, "-n", *base],
                "sudo_noninteractive",
            )
            if privileged["returncode"] == 0:
                return {
                    "ok": True,
                    "feature": feature,
                    "requested_state": state,
                    "method": "sudo_noninteractive",
                    "attempts": attempts,
                }

    last = attempts[-1] if attempts else {}
    error = (
        last.get("stderr")
        or last.get("stdout")
        or "Unable to change offload state"
    )
    return {
        "ok": False,
        "feature": feature,
        "requested_state": state,
        "method": None,
        "error": error,
        "attempts": attempts,
    }


class CaptureOffloadManager:
    """
    Disable capture-distorting NIC offloads before packet capture, verify the
    resulting state, and restore exactly the settings changed by this process.

    The whole Python experiment is never elevated. If a direct ethtool change
    is denied, the manager may retry only the single ethtool command through
    ``sudo -n``. No password prompt is opened by the experiment process.
    """

    def __init__(
        self,
        interface: str,
        *,
        enabled: bool = True,
        required: bool = True,
        allow_sudo_noninteractive: bool = True,
        restore_on_exit: bool = True,
        features: Iterable[str] = ("gro", "gso", "tso", "lro"),
        state_path: str | Path | None = None,
    ) -> None:
        self.interface = str(interface).strip()
        self.enabled = bool(enabled)
        self.required = bool(required)
        self.allow_sudo_noninteractive = bool(
            allow_sudo_noninteractive
        )
        self.restore_on_exit = bool(restore_on_exit)
        self.features = tuple(
            dict.fromkeys(str(item).strip().lower() for item in features)
        )
        self.state_path = (
            Path(state_path)
            if state_path is not None
            else None
        )

        unknown = [
            item
            for item in self.features
            if item not in OFFLOAD_FEATURES
        ]
        if unknown:
            raise CaptureOffloadError(
                "Unknown offload feature(s): "
                + ", ".join(unknown)
            )

        self.before: Dict[str, Any] = {}
        self.after_disable: Dict[str, Any] = {}
        self.after_restore: Dict[str, Any] = {}
        self.disable_actions: list[Dict[str, Any]] = []
        self.restore_actions: list[Dict[str, Any]] = []
        self.changed_features: list[str] = []
        self.started = False
        self.restored = False
        self.status = "not_started"

    def _write_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.state_path.write_text(
            json.dumps(
                self.report(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _enabled_requested_features(
        self,
        state: Dict[str, Any],
    ) -> list[str]:
        result = []
        feature_state = state.get("features", {}) or {}
        for feature in self.features:
            item = feature_state.get(feature, {}) or {}
            if item.get("enabled") is True:
                result.append(feature)
        return result

    def start(self) -> Dict[str, Any]:
        if self.started:
            return self.report()

        self.started = True

        if not self.enabled:
            self.before = read_offload_state(
                self.interface
            )
            self.after_disable = dict(self.before)
            self.status = "management_disabled"
            self._write_state()
            return self.report()

        self.before = read_offload_state(
            self.interface
        )
        if not self.before.get("read_ok"):
            self.status = "preflight_failed"
            self._write_state()
            if self.required:
                raise CaptureOffloadError(
                    "Cannot verify capture-interface offloads on "
                    f"{self.interface!r}: "
                    f"{self.before.get('error', 'unknown ethtool error')}. "
                    "Install ethtool and verify the interface name before "
                    "collecting packet-size fingerprints."
                )
            return self.report()

        before_features = (
            self.before.get("features", {}) or {}
        )

        for feature in self.features:
            item = before_features.get(feature, {}) or {}

            # Unsupported/off features need no mutation.
            if not item.get("supported"):
                continue
            if item.get("enabled") is not True:
                continue

            if item.get("fixed") is True:
                self.disable_actions.append(
                    {
                        "ok": False,
                        "feature": feature,
                        "requested_state": "off",
                        "method": None,
                        "error": (
                            "Driver reports this feature as [fixed]"
                        ),
                        "attempts": [],
                    }
                )
                continue

            action = _run_set_command(
                interface=self.interface,
                feature=feature,
                state="off",
                allow_sudo_noninteractive=(
                    self.allow_sudo_noninteractive
                ),
            )
            self.disable_actions.append(action)
            if action.get("ok"):
                self.changed_features.append(feature)

        self.after_disable = read_offload_state(
            self.interface
        )
        remaining = self._enabled_requested_features(
            self.after_disable
        )

        if (
            not self.after_disable.get("read_ok")
            or remaining
        ):
            self.status = "disable_verification_failed"
            self._write_state()

            # We changed host networking but did not achieve a verified safe
            # capture state. Put back what we changed before failing.
            self.restore(force=True)

            if self.required:
                if remaining:
                    details = ", ".join(remaining)
                    reason = (
                        "the following capture-affecting offloads remain "
                        f"enabled: {details}"
                    )
                else:
                    reason = (
                        "the post-change offload state could not be verified"
                    )
                raise CaptureOffloadError(
                    f"Capture aborted because {reason}. "
                    "The experiment will not collect packet-size data with "
                    "an unverified offload state. If permission was denied, "
                    "install the narrow ethtool sudo rule documented in "
                    "OFFLOAD_PRIVILEGES.md."
                )

            return self.report()

        self.status = "capture_safe"
        self._write_state()
        return self.report()

    def restore(
        self,
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        if self.restored:
            return self.report()

        if not self.started:
            return self.report()

        if (
            not self.restore_on_exit
            and not force
        ):
            self.status = "restore_disabled"
            self.restored = True
            self._write_state()
            return self.report()

        # Only restore features that this manager actually changed. Settings
        # that were already off before the experiment remain off.
        before_features = (
            self.before.get("features", {}) or {}
        )
        for feature in self.changed_features:
            original = (
                before_features.get(feature, {}) or {}
            )
            original_state = (
                "on"
                if original.get("enabled") is True
                else "off"
            )
            action = _run_set_command(
                interface=self.interface,
                feature=feature,
                state=original_state,
                allow_sudo_noninteractive=(
                    self.allow_sudo_noninteractive
                ),
            )
            self.restore_actions.append(action)

        self.after_restore = read_offload_state(
            self.interface
        )
        restore_failures = [
            action
            for action in self.restore_actions
            if not action.get("ok")
        ]

        mismatch = []
        if self.after_restore.get("read_ok"):
            final_features = (
                self.after_restore.get("features", {}) or {}
            )
            for feature in self.changed_features:
                original = (
                    before_features.get(feature, {}) or {}
                )
                final = (
                    final_features.get(feature, {}) or {}
                )
                if (
                    original.get("enabled")
                    is not final.get("enabled")
                ):
                    mismatch.append(feature)
        elif self.changed_features:
            mismatch = list(self.changed_features)

        self.restored = True
        if restore_failures or mismatch:
            self.status = "restore_failed"
        elif self.status != "management_disabled":
            self.status = "restored"

        self._write_state()
        return self.report()

    def report(self) -> Dict[str, Any]:
        return {
            "interface": self.interface,
            "enabled": self.enabled,
            "required": self.required,
            "allow_sudo_noninteractive": (
                self.allow_sudo_noninteractive
            ),
            "restore_on_exit": self.restore_on_exit,
            "managed_features": list(self.features),
            "status": self.status,
            "started": self.started,
            "restored": self.restored,
            "changed_features": list(
                self.changed_features
            ),
            "before": self.before,
            "disable_actions": self.disable_actions,
            "after_disable": self.after_disable,
            "restore_actions": self.restore_actions,
            "after_restore": self.after_restore,
        }
