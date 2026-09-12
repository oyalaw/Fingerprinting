from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai_fingerprint.traffic.analysis import (
    PacketRecord,
    extract_multiscale_feature_rows,
    write_feature_csv,
)


WINDOW_SIZES = (0.5, 1.0, 2.0, 5.0)
MIN_CANONICAL_PACKETS = 20


class CanonicalizationError(RuntimeError):
    pass


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tls_lengths(value: Any) -> tuple[int, ...]:
    raw = str(value or "").replace(",", ";")
    result = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            result.append(int(float(part)))
        except ValueError:
            continue
    return tuple(result)


def _read_ground_truth_registration_state(
    run_root: Path,
    run_id: str,
):
    registered = defaultdict(set)
    confirmed = defaultdict(set)
    clients = set()

    for path in sorted(run_root.rglob("*_ground_truth.jsonl")):
        latest_by_client = {}

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue

                record = json.loads(line)

                record_run = str(
                    record.get("run_id")
                    or record.get("experiment_id", "")
                ).strip()

                if record_run != run_id:
                    continue

                role = str(record.get("role", "")).strip().lower()
                client_id = str(
                    record.get("client_id", "")
                ).strip()

                if role == "client" and client_id:
                    clients.add(client_id)

                event = str(record.get("event", "")).strip()

                if event not in {
                    "network_registration",
                    "network_registration_confirmed",
                }:
                    continue

                if not client_id:
                    continue

                ip = str(record.get("local_ip", "")).strip()
                port = _to_int(record.get("local_port"))

                connection = None
                if ip and port > 0:
                    connection = (ip, port)

                if event == "network_registration":
                    if connection:
                        registered[client_id].add(connection)
                        latest_by_client[client_id] = connection

                elif event == "network_registration_confirmed":
                    if connection is None:
                        connection = latest_by_client.get(client_id)

                    if connection:
                        confirmed[client_id].add(connection)

    return clients, registered, confirmed


def _manifest_entries(run_root: Path, run_id: str):
    entries = []
    packet_counts = defaultdict(int)

    manifest_paths = sorted(
        (
            run_root
            / "proxy"
            / "capture_artifacts"
        ).glob("chunk_*/*_manifest.json")
    )

    for chunk_number, manifest_path in enumerate(
        manifest_paths,
        start=1,
    ):
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise CanonicalizationError(
                f"Cannot read manifest {manifest_path}: {exc}"
            ) from exc

        manifest_run_id = str(
            manifest.get("run_id")
            or manifest.get("experiment_id", "")
        ).strip()

        if manifest_run_id != run_id:
            continue

        capture = manifest.get("capture", {}) or {}
        capture_start_epoch = _to_float(
            capture.get("capture_start_epoch"),
            default=float("nan"),
        )

        if capture_start_epoch != capture_start_epoch:
            raise CanonicalizationError(
                f"{manifest_path} has no capture_start_epoch"
            )

        per_client = (
            manifest.get("outputs", {})
            .get("per_client", {})
            or {}
        )

        if not isinstance(per_client, dict):
            continue

        for alias, item in per_client.items():
            if not isinstance(item, dict):
                continue

            ip = str(item.get("client_ip", "")).strip()
            port = _to_int(item.get("client_port"))

            if not ip or port <= 0:
                continue

            connection = (ip, port)

            packet_count = _to_int(
                item.get("packet_count")
            )
            packet_counts[connection] += packet_count

            declared = str(
                item.get("fingerprint_sequence_csv", "")
            ).strip()

            if declared:
                safe_name = Path(declared).name
            else:
                safe_name = (
                    f"{run_id}__{alias}"
                    "_fingerprint_sequence.csv"
                )

            safe_path = manifest_path.parent / safe_name

            if not safe_path.exists():
                raise CanonicalizationError(
                    "Classifier-safe sequence missing: "
                    f"{safe_path}"
                )

            trace_offset = _to_float(
                item.get("trace_start_offset_sec")
            )

            entries.append(
                {
                    "chunk_number": chunk_number,
                    "manifest": str(manifest_path),
                    "alias": str(alias),
                    "connection": connection,
                    "client_ip": ip,
                    "client_port": port,
                    "packet_count": packet_count,
                    "safe_path": safe_path,
                    "base_epoch": (
                        capture_start_epoch
                        + trace_offset
                    ),
                }
            )

    return entries, packet_counts


def _choose_connections(
    clients,
    registered,
    confirmed,
    packet_counts,
):
    """
    Resolve logical FL clients to observed proxy TCP connections.

    A logical client is anchored by its ground-truth network registration,
    but a reconnect may change the client's source port while retaining the
    same client IP.  Therefore confirmed/registered (IP, port) tuples are
    used to establish client identity, then substantive observed connections
    from the same uniquely-owned client IP are stitched into that client's
    canonical trace.

    Extremely small same-IP connection fragments are treated as setup/retry
    connections rather than independent client traces.
    """
    selected = {}
    reasons = {}

    # First establish the authoritative registration seed for every client.
    seeds_by_client = {}
    base_reason = {}

    for client_id in sorted(clients):
        confirmed_connections = set(
            confirmed.get(client_id, set())
        )

        if confirmed_connections:
            seeds_by_client[client_id] = confirmed_connections
            base_reason[client_id] = (
                "network_registration_confirmed"
            )
            continue

        candidates = set(
            registered.get(client_id, set())
        )

        if not candidates:
            raise CanonicalizationError(
                f"{client_id} has no network registration"
            )

        # Preserve the previous legacy behavior for determining the
        # authoritative connection, then permit same-IP reconnect expansion.
        best = max(
            candidates,
            key=lambda conn: packet_counts.get(conn, 0),
        )

        seeds_by_client[client_id] = {best}
        base_reason[client_id] = (
            "legacy_longest_registered_connection"
        )

    # Determine whether a registered IP uniquely identifies one FL client.
    # Reconnect expansion is safe only when the IP is not claimed by
    # multiple logical clients.
    ip_owners = {}

    for client_id, seeds in seeds_by_client.items():
        for connection in seeds:
            if not connection:
                continue
            ip = str(connection[0]).strip()
            if not ip:
                continue
            ip_owners.setdefault(ip, set()).add(client_id)

    observed_connections = {
        connection
        for connection, count in packet_counts.items()
        if int(count or 0) > 0
    }

    for client_id in sorted(clients):
        seeds = seeds_by_client[client_id]
        resolved = set()
        reconnect_expanded = False
        retry_fragments_removed = False

        for seed in sorted(seeds):
            ip = str(seed[0]).strip()

            # If the IP is ambiguous across logical clients, retain exact
            # ground-truth registration rather than making an unsafe guess.
            if len(ip_owners.get(ip, set())) != 1:
                resolved.add(seed)
                continue

            same_ip = sorted(
                connection
                for connection in observed_connections
                if str(connection[0]).strip() == ip
            )

            if not same_ip:
                resolved.add(seed)
                continue

            if len(same_ip) > 1:
                reconnect_expanded = True

            largest = max(
                int(packet_counts.get(connection, 0) or 0)
                for connection in same_ip
            )

            substantive = set()

            for connection in same_ip:
                count = int(
                    packet_counts.get(connection, 0) or 0
                )

                # Conservative setup/retry filter:
                #
                # Drop only fragments that are BOTH:
                #   * smaller than 100 packets, and
                #   * smaller than 0.01% of the largest same-IP connection.
                #
                # This removes tiny handshake/retry connections such as the
                # 15-packet bert_tiny fragment while preserving legitimate
                # reconnect segments.
                tiny_retry = (
                    count < 100
                    and largest > 0
                    and count < (largest * 0.0001)
                )

                if tiny_retry:
                    retry_fragments_removed = True
                    continue

                substantive.add(connection)

            # Never lose the client merely because all observations happen
            # to be small.  Fall back to the largest observed connection.
            if not substantive:
                substantive.add(
                    max(
                        same_ip,
                        key=lambda conn: packet_counts.get(
                            conn, 0
                        ),
                    )
                )

            resolved.update(substantive)

        if not resolved:
            raise CanonicalizationError(
                f"{client_id} has no resolvable proxy connection"
            )

        selected[client_id] = resolved

        reason = base_reason[client_id]

        if reconnect_expanded:
            reason += "_same_ip_reconnects"

        if retry_fragments_removed:
            reason += "_retry_fragments_removed"

        reasons[client_id] = reason

    return selected, reasons


def _read_safe_sequence(
    path: Path,
    *,
    base_epoch: float,
    chunk_number: int,
    connection_number: int,
):
    packets = []

    synthetic_client = (
        f"canonical_client_{connection_number}"
    )
    synthetic_proxy = "canonical_proxy"

    client_port = 10000 + connection_number
    proxy_port = 8080

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for local_index, row in enumerate(
            reader,
            start=1,
        ):
            relative = _to_float(
                row.get("relative_time_sec")
            )

            direction = str(
                row.get("direction", "unknown")
            ).strip().lower()

            if direction == "up":
                src_ip = synthetic_client
                dst_ip = synthetic_proxy
                src_port = client_port
                dst_port = proxy_port
            elif direction == "down":
                src_ip = synthetic_proxy
                dst_ip = synthetic_client
                src_port = proxy_port
                dst_port = client_port
            else:
                src_ip = synthetic_client
                dst_ip = synthetic_proxy
                src_port = client_port
                dst_port = proxy_port

            packet_index = (
                chunk_number * 1_000_000_000
                + connection_number * 10_000_000
                + local_index
            )

            packets.append(
                PacketRecord(
                    index=packet_index,
                    timestamp_epoch=(
                        base_epoch + relative
                    ),
                    frame_length=_to_int(
                        row.get("frame_length")
                    ),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    transport_protocol=str(
                        row.get(
                            "transport_protocol",
                            "OTHER",
                        )
                    ).strip() or "OTHER",
                    tcp_flags_hex=str(
                        row.get(
                            "tcp_flags_hex",
                            "",
                        )
                        or ""
                    ),
                    tcp_syn=_to_int(
                        row.get("tcp_syn")
                    ),
                    tcp_ack=_to_int(
                        row.get("tcp_ack")
                    ),
                    tcp_fin=_to_int(
                        row.get("tcp_fin")
                    ),
                    tcp_rst=_to_int(
                        row.get("tcp_rst")
                    ),
                    retransmission=int(
                        bool(
                            _to_int(
                                row.get(
                                    "retransmission"
                                )
                            )
                        )
                    ),
                    tls_record_lengths=_tls_lengths(
                        row.get(
                            "tls_record_lengths"
                        )
                    ),
                    direction=direction,
                )
            )

    return packets


def _read_run_metadata(run_root: Path, run_id: str) -> dict[str, Any]:
    """Read descriptive workload metadata for audit only.

    These fields never enter predictor X. They are used to prove that every
    coordinated run, including variants sharing human names such as exp1, is
    represented separately by its unique run_id.
    """
    preferred = []
    for role in ("server", "proxy"):
        role_root = run_root / role
        if role_root.exists():
            preferred.extend(sorted(role_root.rglob("experiment_manifest.json")))
    if not preferred:
        preferred = sorted(run_root.rglob("experiment_manifest.json"))

    for path in preferred:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload_run_id = str(payload.get("run_id") or "").strip()
        if payload_run_id and payload_run_id != run_id:
            continue
        return {
            key: payload.get(key)
            for key in (
                "run_id",
                "experiment_id",
                "storage_locator",
                "family",
                "architecture",
                "variant",
                "application",
                "dataset",
                "framework",
                "deployment",
            )
            if payload.get(key) not in {None, ""}
        }
    return {"run_id": run_id}


def canonicalize_collected_proxy(
    collected_root: Path,
    valid_run_ids,
    output_dir: Path,
):
    collected_root = Path(collected_root)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_files = []
    audit_runs = []
    total_overall = 0

    for run_id in valid_run_ids:
        run_root = collected_root / run_id

        (
            clients,
            registered,
            confirmed,
        ) = _read_ground_truth_registration_state(
            run_root,
            run_id,
        )

        if not clients:
            raise CanonicalizationError(
                f"{run_id}: no client ground truth"
            )

        entries, packet_counts = _manifest_entries(
            run_root,
            run_id,
        )

        if not entries:
            raise CanonicalizationError(
                f"{run_id}: no per-client capture "
                "manifest entries found"
            )

        selected, reasons = _choose_connections(
            clients,
            registered,
            confirmed,
            packet_counts,
        )

        run_audit = {
            "run_id": run_id,
            "metadata": _read_run_metadata(run_root, run_id),
            "clients": [],
        }

        for client_id in sorted(clients):
            wanted = selected[client_id]

            relevant = [
                entry
                for entry in entries
                if entry["connection"] in wanted
            ]

            if not relevant:
                raise CanonicalizationError(
                    f"{run_id}/{client_id}: "
                    "selected FL connection has no "
                    "captured fingerprint sequence"
                )

            connection_numbers = {
                connection: number
                for number, connection in enumerate(
                    sorted(wanted),
                    start=1,
                )
            }

            packets = []

            for entry in relevant:
                packets.extend(
                    _read_safe_sequence(
                        entry["safe_path"],
                        base_epoch=entry["base_epoch"],
                        chunk_number=entry[
                            "chunk_number"
                        ],
                        connection_number=(
                            connection_numbers[
                                entry["connection"]
                            ]
                        ),
                    )
                )

            packets.sort(
                key=lambda packet: (
                    packet.timestamp_epoch,
                    packet.index,
                )
            )

            if len(packets) < MIN_CANONICAL_PACKETS:
                raise CanonicalizationError(
                    f"{run_id}/{client_id}: only "
                    f"{len(packets)} canonical packets; "
                    "likely a setup/retry trace"
                )

            rows = extract_multiscale_feature_rows(
                packets=packets,
                experiment_id=run_id,
                burst_gap_sec=0.05,
                idle_threshold_sec=0.5,
                window_sizes_sec=WINDOW_SIZES,
                packet_information_threshold=2,
            )

            duration = max(
                0.0,
                packets[-1].timestamp_epoch
                - packets[0].timestamp_epoch,
            )

            canonical_rows = []

            for row in rows:
                start = _to_float(
                    row.get("window_start_sec")
                )
                end = _to_float(
                    row.get("window_end_sec")
                )

                enriched = {
                    "experiment_id": run_id,
                    # Use the actual federated client ID only as
                    # grouping metadata. The dataset builder excludes
                    # this field from predictor X.
                    "client_capture_id": client_id,
                    "trace_start_offset_sec": 0.0,
                    "trace_end_offset_sec": duration,
                    "window_start_global_sec": start,
                    "window_end_global_sec": end,
                }

                enriched.update(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "experiment_id"
                    }
                )

                canonical_rows.append(enriched)

            overall_count = sum(
                1
                for row in canonical_rows
                if row.get("row_type") == "overall"
            )

            if overall_count != 1:
                raise CanonicalizationError(
                    f"{run_id}/{client_id}: expected "
                    f"exactly one overall row, got "
                    f"{overall_count}"
                )

            target = (
                output_dir
                / run_id
                / f"{run_id}__{client_id}"
                "_canonical_features.csv"
            )

            write_feature_csv(
                canonical_rows,
                target,
            )

            feature_files.append(str(target))
            total_overall += 1

            run_audit["clients"].append(
                {
                    "client_id": client_id,
                    "selection_reason": (
                        reasons[client_id]
                    ),
                    "selected_connections": [
                        {
                            "client_ip": ip,
                            "client_port": port,
                        }
                        for ip, port in sorted(
                            wanted
                        )
                    ],
                    "source_sequence_files": [
                        str(entry["safe_path"])
                        for entry in relevant
                    ],
                    "source_chunk_count": len(
                        {
                            entry["chunk_number"]
                            for entry in relevant
                        }
                    ),
                    "packet_count": len(packets),
                    "overall_rows": overall_count,
                    "feature_rows": len(
                        canonical_rows
                    ),
                    "output": str(target),
                }
            )

        audit_runs.append(run_audit)

    audit = {
        "policy": {
            "one_overall_per_run_client": True,
            "confirmed_connections_preferred": True,
            "legacy_retry_fallback": (
                "longest_registered_connection"
            ),
            "rotation_chunks_stitched": True,
            "window_source": (
                "recomputed from canonical stitched "
                "packet sequence"
            ),
            "live_features_used_for_training": False,
            "window_sizes_sec": list(
                WINDOW_SIZES
            ),
        },
        "run_count": len(valid_run_ids),
        "run_ids": [str(value) for value in valid_run_ids],
        "overall_sample_count": total_overall,
        "feature_file_count": len(
            feature_files
        ),
        "feature_files": feature_files,
        "runs": audit_runs,
    }

    audit_path = (
        output_dir
        / "canonicalization_audit.json"
    )

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "feature_files": feature_files,
        "feature_file_count": len(
            feature_files
        ),
        "overall_sample_count": (
            total_overall
        ),
        "run_count": len(valid_run_ids),
        "run_ids": [str(value) for value in valid_run_ids],
        "audit_json": str(audit_path),
    }


def main():
    root = Path(
        "collected_experiments"
    ).resolve()

    validation_path = (
        root / "collection_validation.json"
    )

    if not validation_path.exists():
        raise SystemExit(
            "Run run_central_fingerprinting.py "
            "once first so collection_validation.json exists."
        )

    validation = json.loads(
        validation_path.read_text(
            encoding="utf-8"
        )
    )

    result = canonicalize_collected_proxy(
        collected_root=root,
        valid_run_ids=validation[
            "valid_run_ids"
        ],
        output_dir=(
            Path("fingerprinting_dataset")
            / "canonical_proxy_features"
        ),
    )

    print("\nCanonicalization complete:")
    for key, value in result.items():
        if key != "feature_files":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
