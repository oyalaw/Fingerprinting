from __future__ import annotations

import datetime as dt
import json
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict


COORDINATION_PROTOCOL_VERSION = "1.0"
DEFAULT_COORDINATION_PORT = 8081


class ExperimentCoordinationError(RuntimeError):
    pass


def generate_run_id() -> str:
    """Return a neutral per-execution identifier containing no AI labels."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    return f"run_{stamp}_{secrets.token_hex(4)}"


def ensure_run_id(config: Dict[str, Any]) -> str:
    experiment = config.setdefault("experiment", {})
    value = str(experiment.get("run_id") or "").strip()
    if not value:
        value = generate_run_id()
        experiment["run_id"] = value
    return value


@dataclass(frozen=True)
class ActiveRunInfo:
    run_id: str
    host: str
    port: int
    protocol_version: str = COORDINATION_PROTOCOL_VERSION


class ActiveRunCoordinator:
    """Tiny out-of-band control endpoint that exposes only the neutral run ID.

    It intentionally does not return family, architecture, variant, application,
    dataset, framework, storage locator, or any other ground-truth label.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        coordination = config.get("coordination", {}) or {}
        node = config.get("node", {}) or {}
        self.host = str(coordination.get("host") or node.get("host") or "0.0.0.0")
        self.port = int(coordination.get("port", DEFAULT_COORDINATION_PORT))
        self.run_id = ensure_run_id(config)
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        self.port = int(listener.getsockname()[1])
        listener.listen(8)
        listener.settimeout(0.5)
        self._listener = listener
        self._thread = threading.Thread(
            target=self._serve,
            name="aifp-active-run-coordinator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        self._listener = None

    def _serve(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                conn, _peer = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(2.0)
                try:
                    raw = b""
                    while b"\n" not in raw and len(raw) < 4096:
                        chunk = conn.recv(1024)
                        if not chunk:
                            break
                        raw += chunk
                    request = json.loads(raw.split(b"\n", 1)[0].decode("utf-8"))
                    if request.get("op") != "active_run":
                        response = {
                            "status": "error",
                            "error": "unsupported operation",
                            "protocol_version": COORDINATION_PROTOCOL_VERSION,
                        }
                    else:
                        response = {
                            "status": "ok",
                            "run_id": self.run_id,
                            "protocol_version": COORDINATION_PROTOCOL_VERSION,
                        }
                except Exception as exc:
                    response = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "protocol_version": COORDINATION_PROTOCOL_VERSION,
                    }
                try:
                    conn.sendall((json.dumps(response, sort_keys=True) + "\n").encode("utf-8"))
                except OSError:
                    pass


def query_active_run(
    host: str,
    port: int = DEFAULT_COORDINATION_PORT,
    *,
    timeout_sec: float = 2.0,
) -> ActiveRunInfo:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_sec) as sock:
            sock.settimeout(timeout_sec)
            sock.sendall(
                (json.dumps({"op": "active_run", "protocol_version": COORDINATION_PROTOCOL_VERSION}) + "\n").encode("utf-8")
            )
            raw = b""
            while b"\n" not in raw and len(raw) < 4096:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                raw += chunk
    except OSError as exc:
        raise ExperimentCoordinationError(
            f"Could not reach experiment coordinator at {host}:{port}: {exc}"
        ) from exc

    if not raw:
        raise ExperimentCoordinationError("Experiment coordinator returned no response")
    try:
        payload = json.loads(raw.split(b"\n", 1)[0].decode("utf-8"))
    except Exception as exc:
        raise ExperimentCoordinationError("Experiment coordinator returned invalid JSON") from exc
    if payload.get("status") != "ok":
        raise ExperimentCoordinationError(str(payload.get("error") or "coordinator error"))
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in run_id):
        raise ExperimentCoordinationError("Experiment coordinator returned an invalid neutral run ID")
    return ActiveRunInfo(
        run_id=run_id,
        host=str(host),
        port=int(port),
        protocol_version=str(payload.get("protocol_version") or COORDINATION_PROTOCOL_VERSION),
    )


def wait_for_active_run(
    host: str,
    port: int = DEFAULT_COORDINATION_PORT,
    *,
    retry_interval_sec: float = 2.0,
    announce_interval_sec: float = 10.0,
) -> ActiveRunInfo:
    """Wait until the server coordinator is available; Ctrl-C remains the escape."""
    last_announce = 0.0
    while True:
        try:
            return query_active_run(host, port)
        except ExperimentCoordinationError as exc:
            now = time.monotonic()
            if now - last_announce >= announce_interval_sec:
                print(
                    f"Waiting for experiment coordinator at {host}:{port}... "
                    f"({exc})"
                )
                last_announce = now
            time.sleep(max(0.2, float(retry_interval_sec)))
