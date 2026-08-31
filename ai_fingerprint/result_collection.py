from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence

import yaml

from .experiment_integrity import atomic_write_json, sha256_file, utc_now_iso
from .metadata import output_role_token


COLLECTION_PROTOCOL_VERSION = "1.0"
DEFAULT_COLLECTION_PORT = 8090
ANALYSIS_SUFFIXES = {
    ".csv", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".log"
}
EXCLUDED_NAMES = {
    "collection_receipt.json",
    "collection_status.json",
    "collection_manifest.json",
}
EXCLUDED_SUFFIXES = {
    ".pcap", ".pcapng", ".cap", ".npz", ".npy", ".pt", ".pth",
    ".onnx", ".tflite", ".engine", ".zip", ".tar", ".gz",
}


class ResultCollectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionReceipt:
    run_id: str
    participant: str
    archive_sha256: str
    received_at_utc: str
    collector_path: str
    file_count: int
    total_uncompressed_bytes: int


def _safe_token(value: Any, field: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ResultCollectionError(f"{field} must not be empty")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in raw):
        raise ResultCollectionError(
            f"{field} contains unsupported characters: {raw!r}"
        )
    return raw


def collection_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    value = dict(config.get("result_collection", {}) or {})
    value.setdefault("enabled", True)
    value.setdefault("collector_host", "10.42.0.195")
    value.setdefault("collector_port", DEFAULT_COLLECTION_PORT)
    value.setdefault("timeout_sec", 10.0)
    value.setdefault("retry_attempts", 3)
    value.setdefault("retry_delay_sec", 1.0)
    value.setdefault("shared_token", "")
    value.setdefault("max_archive_mb", 1024)
    value.setdefault("central_root", "collected_experiments")
    return value


def participant_token(config: Mapping[str, Any]) -> str:
    role = str(config.get("node", {}).get("role", "")).strip().lower()
    if not role and "proxy" in config:
        role = "proxy"
    if role == "client":
        return _safe_token(output_role_token(dict(config)), "participant")
    if role in {"server", "proxy"}:
        return role
    raise ResultCollectionError(f"Unsupported result-collection role: {role!r}")


def _analysis_files(output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir)
        if any(part.lower() in {"checkpoints", "pcap", "_tls"} for part in rel.parts):
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        suffix = path.suffix.lower()
        if suffix in EXCLUDED_SUFFIXES:
            continue
        if suffix not in ANALYSIS_SUFFIXES:
            continue
        files.append(path)
    return sorted(files)


def build_collection_archive(
    config: Mapping[str, Any],
    *,
    temp_dir: str | Path | None = None,
) -> tuple[Path, Dict[str, Any]]:
    experiment = config.get("experiment", {}) or {}
    run_id = _safe_token(experiment.get("run_id"), "run_id")
    participant = participant_token(config)
    output_dir = Path(str(experiment.get("output_dir", "")))
    if not output_dir.exists():
        raise ResultCollectionError(f"Local result directory not found: {output_dir}")

    selected = _analysis_files(output_dir)
    if not selected:
        raise ResultCollectionError(
            f"No analysis-relevant files found under {output_dir}"
        )

    file_entries = []
    total_bytes = 0
    for path in selected:
        size = path.stat().st_size
        total_bytes += size
        file_entries.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )

    content_identity = {
        "protocol_version": COLLECTION_PROTOCOL_VERSION,
        "run_id": run_id,
        "participant": participant,
        "files": file_entries,
    }
    content_sha256 = hashlib.sha256(
        json.dumps(
            content_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest: Dict[str, Any] = {
        "protocol_version": COLLECTION_PROTOCOL_VERSION,
        "created_at_utc": utc_now_iso(),
        "run_id": run_id,
        "experiment_id": experiment.get("experiment_id"),
        "storage_locator": experiment.get("storage_locator"),
        "participant": participant,
        "role": participant if participant in {"server", "proxy"} else "client",
        "client_id": (
            config.get("federated", {}).get("client_id")
            if participant.startswith("client")
            else None
        ),
        "local_result_retained": True,
        "source_output_dir": str(output_dir.resolve()),
        "file_count": len(file_entries),
        "total_uncompressed_bytes": total_bytes,
        "content_sha256": content_sha256,
        "files": file_entries,
    }

    temp_root = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"aifp_{run_id}_{participant}_",
        suffix=".zip",
        dir=str(temp_root),
    )
    os.close(fd)
    archive = Path(tmp_name)
    try:
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            zf.writestr(
                "collection_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True),
            )
            for path in selected:
                zf.write(path, path.relative_to(output_dir).as_posix())
        archive_sha = sha256_file(archive)
        manifest["archive_sha256"] = archive_sha
        return archive, manifest
    except Exception:
        archive.unlink(missing_ok=True)
        raise


def _write_collection_status(
    output_dir: Path,
    *,
    status: str,
    run_id: str,
    participant: str,
    collector_host: str,
    collector_port: int,
    error: str | None = None,
    receipt: Mapping[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "status": status,
        "run_id": run_id,
        "participant": participant,
        "collector_host": collector_host,
        "collector_port": int(collector_port),
        "local_results_retained": True,
        "timestamp_utc": utc_now_iso(),
    }
    if error:
        payload["error"] = error
    if receipt:
        payload["receipt"] = dict(receipt)
    atomic_write_json(output_dir / "collection_status.json", payload)


def upload_result_copy(config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = collection_config(config)
    if not bool(cfg.get("enabled", True)):
        return {"status": "DISABLED"}

    experiment = config.get("experiment", {}) or {}
    run_id = _safe_token(experiment.get("run_id"), "run_id")
    participant = participant_token(config)
    output_dir = Path(str(experiment.get("output_dir", "")))
    host = str(cfg.get("collector_host") or "").strip()
    port = int(cfg.get("collector_port", DEFAULT_COLLECTION_PORT))
    if not host:
        raise ResultCollectionError("result_collection.collector_host is required")

    _write_collection_status(
        output_dir,
        status="TRANSFERRING",
        run_id=run_id,
        participant=participant,
        collector_host=host,
        collector_port=port,
    )

    archive, manifest = build_collection_archive(config)
    archive_sha = str(manifest["archive_sha256"])
    archive_size = archive.stat().st_size
    max_bytes = int(float(cfg.get("max_archive_mb", 1024)) * 1024 * 1024)
    if archive_size > max_bytes:
        archive.unlink(missing_ok=True)
        raise ResultCollectionError(
            f"Collection archive is {archive_size} bytes, above configured "
            f"limit {max_bytes}. Raw PCAP/model artifacts should not be collected."
        )

    attempts = max(int(cfg.get("retry_attempts", 3)), 1)
    timeout_sec = max(float(cfg.get("timeout_sec", 10.0)), 1.0)
    delay = max(float(cfg.get("retry_delay_sec", 1.0)), 0.0)
    token = str(cfg.get("shared_token") or "")
    last_error: Exception | None = None

    try:
        for attempt in range(1, attempts + 1):
            conn: http.client.HTTPConnection | None = None
            try:
                conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
                headers = {
                    "Content-Type": "application/zip",
                    "Content-Length": str(archive_size),
                    "X-AIFP-Protocol": COLLECTION_PROTOCOL_VERSION,
                    "X-AIFP-Run-ID": run_id,
                    "X-AIFP-Participant": participant,
                    "X-AIFP-Archive-SHA256": archive_sha,
                    "X-AIFP-Content-SHA256": str(manifest["content_sha256"]),
                }
                if token:
                    headers["X-AIFP-Token"] = token
                with archive.open("rb") as handle:
                    conn.request("POST", "/v1/result", body=handle, headers=headers)
                    response = conn.getresponse()
                    body = response.read().decode("utf-8", errors="replace")
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    payload = {"message": body}
                if response.status != 200:
                    raise ResultCollectionError(
                        f"Collector returned HTTP {response.status}: "
                        f"{payload.get('error') or payload.get('message') or body}"
                    )
                payload.setdefault("status", "VERIFIED")
                payload["local_results_retained"] = True
                atomic_write_json(output_dir / "collection_receipt.json", payload)
                _write_collection_status(
                    output_dir,
                    status="VERIFIED",
                    run_id=run_id,
                    participant=participant,
                    collector_host=host,
                    collector_port=port,
                    receipt=payload,
                )
                return payload
            except Exception as exc:
                last_error = exc
                if attempt < attempts and delay > 0:
                    time.sleep(delay)
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
        raise ResultCollectionError(str(last_error or "collection upload failed"))
    finally:
        archive.unlink(missing_ok=True)


def auto_upload_result_copy(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Best-effort copy. Collection failure never changes experiment validity."""
    cfg = collection_config(config)
    if not bool(cfg.get("enabled", True)):
        return {"status": "DISABLED"}
    experiment = config.get("experiment", {}) or {}
    if not str(experiment.get("run_id") or "").strip():
        return {"status": "SKIPPED", "reason": "no coordinated run_id"}
    output_dir = Path(str(experiment.get("output_dir", "")))
    try:
        receipt = upload_result_copy(config)
        print(
            "[collection] verified central copy: "
            f"{receipt.get('run_id')}/{receipt.get('participant')}"
        )
        return receipt
    except Exception as exc:
        try:
            _write_collection_status(
                output_dir,
                status="PENDING",
                run_id=str(experiment.get("run_id") or "unknown"),
                participant=(participant_token(config) if experiment.get("run_id") else "unknown"),
                collector_host=str(cfg.get("collector_host") or ""),
                collector_port=int(cfg.get("collector_port", DEFAULT_COLLECTION_PORT)),
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            pass
        print(
            "[collection] central replication pending; local results are safe: "
            f"{exc}"
        )
        return {"status": "PENDING", "error": str(exc)}


def _safe_zip_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ResultCollectionError(f"Unsafe archive member: {name!r}")
    return path


def _verify_extracted_manifest(
    extracted_dir: Path,
    *,
    expected_run_id: str,
    expected_participant: str,
) -> Dict[str, Any]:
    manifest_path = extracted_dir / "collection_manifest.json"
    if not manifest_path.exists():
        raise ResultCollectionError("Collection archive has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("protocol_version")) != COLLECTION_PROTOCOL_VERSION:
        raise ResultCollectionError("Unsupported collection protocol version")
    if _safe_token(manifest.get("run_id"), "run_id") != expected_run_id:
        raise ResultCollectionError("Run ID does not match upload header")
    if _safe_token(manifest.get("participant"), "participant") != expected_participant:
        raise ResultCollectionError("Participant does not match upload header")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ResultCollectionError("Collection manifest contains no files")
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ResultCollectionError("Invalid collection manifest entry")
        rel = _safe_zip_member(str(entry.get("path", "")))
        path = extracted_dir.joinpath(*rel.parts)
        if not path.is_file():
            raise ResultCollectionError(f"Collected file missing after extraction: {rel}")
        size = path.stat().st_size
        expected_size = int(entry.get("size_bytes", -1))
        if size != expected_size:
            raise ResultCollectionError(
                f"Size mismatch for {rel}: expected {expected_size}, got {size}"
            )
        digest = sha256_file(path)
        if digest != str(entry.get("sha256", "")):
            raise ResultCollectionError(f"SHA256 mismatch for {rel}")
        total += size
    if total != int(manifest.get("total_uncompressed_bytes", total)):
        raise ResultCollectionError("Collection manifest total byte count mismatch")
    return manifest


def _update_collection_index(root: Path, receipt: Mapping[str, Any]) -> None:
    index_path = root / "collection_index.json"
    current: Dict[str, Any] = {}
    if index_path.exists():
        try:
            current = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    participants = dict(current.get("participants", {}) or {})
    participants[str(receipt["participant"])] = dict(receipt)
    atomic_write_json(
        index_path,
        {
            "run_id": receipt["run_id"],
            "updated_at_utc": utc_now_iso(),
            "participants": participants,
        },
    )


class _CollectorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, root: Path, shared_token: str, max_upload_bytes: int):
        super().__init__(address, handler)
        self.collection_root = root
        self.shared_token = shared_token
        self.max_upload_bytes = max_upload_bytes


class _CollectorHandler(BaseHTTPRequestHandler):
    server_version = "AIFPResultCollector/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print("[collector] " + (format % args))

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        self._json(
            200,
            {
                "status": "ok",
                "protocol_version": COLLECTION_PROTOCOL_VERSION,
                "root": str(self.server.collection_root),  # type: ignore[attr-defined]
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/result":
            self._json(404, {"error": "not found"})
            return
        server: _CollectorHTTPServer = self.server  # type: ignore[assignment]
        if server.shared_token:
            if self.headers.get("X-AIFP-Token", "") != server.shared_token:
                self._json(401, {"error": "invalid collector token"})
                return
        try:
            run_id = _safe_token(self.headers.get("X-AIFP-Run-ID"), "run_id")
            participant = _safe_token(
                self.headers.get("X-AIFP-Participant"), "participant"
            )
            protocol = str(self.headers.get("X-AIFP-Protocol", ""))
            if protocol != COLLECTION_PROTOCOL_VERSION:
                raise ResultCollectionError("Unsupported collection protocol")
            expected_archive_sha = str(
                self.headers.get("X-AIFP-Archive-SHA256", "")
            ).strip().lower()
            expected_content_sha = str(
                self.headers.get("X-AIFP-Content-SHA256", "")
            ).strip().lower()
            if len(expected_archive_sha) != 64:
                raise ResultCollectionError("Invalid archive SHA256 header")
            if len(expected_content_sha) != 64:
                raise ResultCollectionError("Invalid content SHA256 header")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ResultCollectionError("Upload body is empty")
            if length > server.max_upload_bytes:
                self._json(413, {"error": "collection archive exceeds configured limit"})
                return

            root = server.collection_root
            incoming = root / "_incoming"
            incoming.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{run_id}_{participant}_", suffix=".zip", dir=str(incoming)
            )
            os.close(fd)
            tmp_archive = Path(tmp_name)
            remaining = length
            digest = hashlib.sha256()
            with tmp_archive.open("wb") as handle:
                while remaining > 0:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ResultCollectionError("Upload ended before Content-Length")
                    handle.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            actual_sha = digest.hexdigest()
            if actual_sha != expected_archive_sha:
                raise ResultCollectionError("Archive SHA256 mismatch")

            run_root = root / run_id
            participant_dir = run_root / participant
            existing_receipt = participant_dir / "_collection_receipt.json"
            if existing_receipt.exists():
                existing = json.loads(existing_receipt.read_text(encoding="utf-8"))
                if str(existing.get("content_sha256")) == expected_content_sha:
                    tmp_archive.unlink(missing_ok=True)
                    self._json(200, existing)
                    return
                raise ResultCollectionError(
                    "A different verified content set already exists for this run/participant"
                )

            temp_extract: Path | None = Path(
                tempfile.mkdtemp(prefix=f".{participant}.extract.", dir=str(root))
            )
            try:
                with zipfile.ZipFile(tmp_archive, "r") as zf:
                    for info in zf.infolist():
                        rel = _safe_zip_member(info.filename)
                        target = temp_extract.joinpath(*rel.parts)
                        if info.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info, "r") as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)
                manifest = _verify_extracted_manifest(
                    temp_extract,
                    expected_run_id=run_id,
                    expected_participant=participant,
                )
                if str(manifest.get("content_sha256", "")) != expected_content_sha:
                    raise ResultCollectionError("Content SHA256 does not match upload header")
                run_root.mkdir(parents=True, exist_ok=True)
                if participant_dir.exists():
                    raise ResultCollectionError(
                        f"Collector destination already exists: {participant_dir}"
                    )
                os.replace(temp_extract, participant_dir)
                temp_extract = None
                receipt = {
                    "status": "VERIFIED",
                    "protocol_version": COLLECTION_PROTOCOL_VERSION,
                    "run_id": run_id,
                    "participant": participant,
                    "archive_sha256": actual_sha,
                    "content_sha256": expected_content_sha,
                    "received_at_utc": utc_now_iso(),
                    "collector_path": str(participant_dir.resolve()),
                    "file_count": int(manifest.get("file_count", 0)),
                    "total_uncompressed_bytes": int(
                        manifest.get("total_uncompressed_bytes", 0)
                    ),
                    "local_results_retained": True,
                }
                atomic_write_json(participant_dir / "_collection_receipt.json", receipt)
                _update_collection_index(run_root, receipt)
                archives = run_root / "_archives"
                archives.mkdir(parents=True, exist_ok=True)
                archive_target = archives / f"{participant}_{actual_sha[:12]}.zip"
                os.replace(tmp_archive, archive_target)
                self._json(200, receipt)
            finally:
                if temp_extract is not None and temp_extract.exists():
                    shutil.rmtree(temp_extract, ignore_errors=True)
                tmp_archive.unlink(missing_ok=True)
        except ResultCollectionError as exc:
            self._json(409, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def load_collector_settings(path: str | Path = "collector.yaml") -> Dict[str, Any]:
    defaults = {
        "host": "0.0.0.0",
        "port": DEFAULT_COLLECTION_PORT,
        "root": "collected_experiments",
        "shared_token": "",
        "max_upload_mb": 1024,
    }
    config_path = Path(path)
    if not config_path.exists():
        return defaults
    supplied = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if "collector" in supplied and isinstance(supplied["collector"], dict):
        supplied = supplied["collector"]
    result = dict(defaults)
    result.update(dict(supplied))
    return result


def serve_collector_forever(settings: Mapping[str, Any]) -> None:
    host = str(settings.get("host") or "0.0.0.0")
    port = int(settings.get("port", DEFAULT_COLLECTION_PORT))
    root = Path(str(settings.get("root") or "collected_experiments")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = str(settings.get("shared_token") or "")
    max_upload_bytes = int(float(settings.get("max_upload_mb", 1024)) * 1024 * 1024)
    server = _CollectorHTTPServer(
        (host, port),
        _CollectorHandler,
        root=root,
        shared_token=token,
        max_upload_bytes=max_upload_bytes,
    )
    print(f"[collector] listening on {host}:{port}")
    print(f"[collector] central root: {root}")
    if not token:
        print("[collector] warning: shared_token is empty; use only on the isolated research LAN")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[collector] stopping")
    finally:
        server.server_close()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def discover_pending_result_configs(root: str | Path = "experiments") -> list[Path]:
    base = Path(root)
    configs: list[Path] = []
    for status_path in base.rglob("experiment_status.json"):
        output_dir = status_path.parent
        config_path = output_dir / "config_effective.yaml"
        if not config_path.exists():
            config_path = output_dir / "config.yaml"
        if not config_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(status.get("status", "")).upper() in {"RUNNING", ""}:
            continue
        receipt = output_dir / "collection_receipt.json"
        if receipt.exists():
            try:
                if str(json.loads(receipt.read_text(encoding="utf-8")).get("status")) == "VERIFIED":
                    continue
            except Exception:
                pass
        configs.append(config_path)
    return sorted(set(configs))


def resend_pending_results(root: str | Path = "experiments") -> list[Dict[str, Any]]:
    results = []
    for config_path in discover_pending_result_configs(root):
        config = _load_yaml(config_path)
        if not config:
            continue
        result = auto_upload_result_copy(config)
        results.append({"config": str(config_path), **result})
    return results


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def validate_collected_run(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    run_id = run_path.name
    participant_dirs = [
        path for path in run_path.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ] if run_path.exists() else []
    participants = {path.name: path for path in participant_dirs}
    issues: list[str] = []

    server_dir = participants.get("server")
    proxy_dir = participants.get("proxy")
    if server_dir is None:
        issues.append("missing server copy")
    if proxy_dir is None:
        issues.append("missing proxy copy")

    client_names = sorted(name for name in participants if name.startswith("client"))
    expected_clients = None
    if server_dir is not None:
        for config_name in ("config_effective.yaml", "config.yaml"):
            config_path = server_dir / config_name
            if config_path.exists():
                cfg = _load_yaml(config_path)
                try:
                    expected_clients = int(cfg.get("federated", {}).get("expected_clients"))
                except Exception:
                    expected_clients = None
                if expected_clients:
                    break
        if expected_clients is None:
            policy_candidates = list(server_dir.rglob("server_training_policy.json"))
            for policy_path in policy_candidates:
                policy = _read_json(policy_path)
                try:
                    expected_clients = int(policy.get("partition_client_count"))
                except Exception:
                    expected_clients = None
                if expected_clients:
                    break
    if expected_clients is not None and len(client_names) != expected_clients:
        issues.append(
            f"expected {expected_clients} clients but collected {len(client_names)}: {client_names}"
        )
    if not client_names:
        issues.append("no client copies collected")

    for name, path in sorted(participants.items()):
        receipt = path / "_collection_receipt.json"
        if not receipt.exists():
            issues.append(f"{name}: missing collector receipt")
        statuses = list(path.rglob("experiment_status.json"))
        if not statuses:
            issues.append(f"{name}: missing experiment_status.json")
            continue
        terminal = _read_json(statuses[0])
        status = str(terminal.get("status", "")).upper()
        if status != "COMPLETED":
            issues.append(f"{name}: experiment status is {status or 'UNKNOWN'}")
        manifest_run = None
        role_manifest = next(iter(path.rglob("role_manifest.json")), None)
        if role_manifest:
            manifest_run = str(_read_json(role_manifest).get("run_id") or "")
        if manifest_run and manifest_run != run_id:
            issues.append(f"{name}: role_manifest run_id {manifest_run!r} != {run_id!r}")

    if proxy_dir is not None:
        feature_files = [
            path for path in proxy_dir.rglob("*_features.csv")
            if "_X_proxy" not in path.name
        ]
        if not feature_files:
            issues.append("proxy: no extracted *_features.csv")
    ground_truth_files = []
    for client_name in client_names:
        ground_truth_files.extend(participants[client_name].rglob("*_ground_truth.jsonl"))
    if not ground_truth_files:
        issues.append("clients: no *_ground_truth.jsonl files")

    return {
        "run_id": run_id,
        "status": "VALID" if not issues else "PARTIAL",
        "expected_clients": expected_clients,
        "collected_clients": client_names,
        "participants": sorted(participants),
        "issues": issues,
    }


def validate_collected_root(root: str | Path = "collected_experiments") -> Dict[str, Any]:
    base = Path(root)
    runs = []
    if base.exists():
        for path in sorted(base.iterdir()):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            runs.append(validate_collected_run(path))
    valid = [item["run_id"] for item in runs if item["status"] == "VALID"]
    return {
        "root": str(base.resolve()),
        "run_count": len(runs),
        "valid_run_count": len(valid),
        "valid_run_ids": valid,
        "runs": runs,
    }
