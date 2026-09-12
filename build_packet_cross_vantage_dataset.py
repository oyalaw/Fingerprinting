from __future__ import annotations

"""Build packet-budget datasets using the established canonical proxy resolver.

Proxy packet traces are NOT identified from trace_001/trace_002 filenames.
They are resolved exactly as the central fingerprinting pipeline resolves them:
client/server ground-truth network registrations + proxy capture manifests +
confirmed FL connection selection + stitched rotation chunks.

Client/server packet sequences, when present in future runs, are discovered
separately. Endpoint telemetry is never substituted for packet metadata.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from ai_fingerprint.packet_fingerprinting import (
    PACKET_COUNTS,
    PacketObservation,
    canonical_client_label,
    infer_source_role,
    is_safe_sequence_csv,
    iter_packet_budget_samples,
    normalize_client_id,
    read_sequence_groups,
)

# Reuse the same resolver already used by the validated central pipeline.
from canonicalize_central_proxy import (
    MIN_CANONICAL_PACKETS,
    CanonicalizationError,
    _choose_connections,
    _manifest_entries,
    _read_ground_truth_registration_state,
    _read_safe_sequence,
)


LABEL_COLUMNS = ("family", "architecture", "variant", "application")


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def load_labels(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Return one label row per (experiment, client), ignoring repeated windows."""
    labels: dict[tuple[str, str], dict[str, str]] = {}
    conflicts: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            experiment = str(row.get("experiment_id", "")).strip()
            client_raw = (
                row.get("resolved_client_id")
                or row.get("client_capture_id")
                or row.get("client_id")
            )
            client = normalize_client_id(client_raw)
            if not experiment or not client:
                continue
            value = {name: str(row.get(name, "")).strip() for name in LABEL_COLUMNS}
            key = (experiment, client)
            old = labels.get(key)
            if old is None:
                labels[key] = value
            elif old != value:
                conflicts.append({"key": key, "old": old, "new": value})
                if len(conflicts) >= 5:
                    break
    if conflicts:
        raise RuntimeError(
            "Ground-truth label conflicts detected: " + json.dumps(conflicts, indent=2)
        )
    return labels


def _packet_record_to_observation(packet) -> PacketObservation:
    return PacketObservation(
        timestamp=float(packet.timestamp_epoch),
        frame_length=float(packet.frame_length),
        direction=str(packet.direction or "unknown").strip().lower(),
        transport_protocol=str(packet.transport_protocol or "OTHER").strip().upper(),
        tcp_syn=int(bool(packet.tcp_syn)),
        tcp_ack=int(bool(packet.tcp_ack)),
        tcp_fin=int(bool(packet.tcp_fin)),
        tcp_rst=int(bool(packet.tcp_rst)),
        retransmission=int(bool(packet.retransmission)),
        tls_record_lengths=tuple(float(x) for x in packet.tls_record_lengths),
    )


def _label_run_ids(label_map: dict[tuple[str, str], dict[str, str]]) -> list[str]:
    return sorted({run_id for run_id, _client in label_map})


def canonical_proxy_specs(
    collected_root: Path,
    label_map: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve proxy sequence chunks to actual FL clients without reading packets."""
    specs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for run_id in _label_run_ids(label_map):
        run_root = collected_root / run_id
        if not run_root.is_dir():
            skipped.append({
                "run_id": run_id,
                "participant": "proxy",
                "role": "proxy",
                "reason": "collected_run_directory_missing",
            })
            continue
        try:
            clients, registered, confirmed = _read_ground_truth_registration_state(
                run_root, run_id
            )
            if not clients:
                raise CanonicalizationError("no client ground-truth registrations")
            entries, packet_counts = _manifest_entries(run_root, run_id)
            if not entries:
                raise CanonicalizationError("no per-client proxy manifest entries")
            selected, reasons = _choose_connections(
                clients, registered, confirmed, packet_counts
            )
        except Exception as exc:
            skipped.append({
                "run_id": run_id,
                "participant": "proxy",
                "role": "proxy",
                "reason": f"canonical_resolution_error: {exc}",
            })
            continue

        for client_id in sorted(clients):
            label_key = (run_id, normalize_client_id(client_id))
            if label_key not in label_map:
                skipped.append({
                    "run_id": run_id,
                    "participant": "proxy",
                    "role": "proxy",
                    "client_id": client_id,
                    "reason": "no_matching_ground_truth_label",
                })
                continue

            wanted = selected.get(client_id, set())
            relevant = [entry for entry in entries if entry["connection"] in wanted]
            if not relevant:
                skipped.append({
                    "run_id": run_id,
                    "participant": "proxy",
                    "role": "proxy",
                    "client_id": client_id,
                    "reason": "selected_fl_connection_has_no_sequence",
                })
                continue

            packet_count = sum(int(entry.get("packet_count", 0) or 0) for entry in relevant)
            specs.append({
                "run_id": run_id,
                "participant": "proxy",
                "role": "proxy",
                "client_id": client_id,
                "selection_reason": reasons.get(client_id, ""),
                "connections": sorted(wanted),
                "entries": relevant,
                "packet_count_manifest": packet_count,
                "source_files": [str(entry["safe_path"]) for entry in relevant],
                "label": label_map[label_key],
            })

    return specs, skipped


def load_canonical_proxy_packets(spec: dict[str, Any]) -> list[PacketObservation]:
    """Stitch the selected proxy chunks exactly as the central canonicalizer does."""
    wanted = set(tuple(x) for x in spec["connections"])
    connection_numbers = {
        connection: number
        for number, connection in enumerate(sorted(wanted), start=1)
    }
    packet_records = []
    for entry in spec["entries"]:
        packet_records.extend(
            _read_safe_sequence(
                entry["safe_path"],
                base_epoch=entry["base_epoch"],
                chunk_number=entry["chunk_number"],
                connection_number=connection_numbers[entry["connection"]],
            )
        )
    packet_records.sort(key=lambda p: (p.timestamp_epoch, p.index))
    if len(packet_records) < MIN_CANONICAL_PACKETS:
        return []
    return [_packet_record_to_observation(p) for p in packet_records]


def discover_endpoint_sequence_files(collected_root: Path) -> Iterator[tuple[str, str, str, Path]]:
    """Discover only endpoint packet sequences; proxy uses canonical resolver above."""
    for run_dir in sorted(p for p in collected_root.iterdir() if p.is_dir()):
        run_id = run_dir.name
        for participant_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            participant = participant_dir.name
            role = infer_source_role(participant)
            if role not in {"client", "server"}:
                continue
            for path in participant_dir.rglob("*.csv"):
                if is_safe_sequence_csv(path):
                    yield run_id, participant, role, path


def _endpoint_clients(path: Path, role: str, participant: str) -> list[str]:
    from ai_fingerprint.packet_fingerprinting import infer_client_ids_from_file

    inferred = infer_client_ids_from_file(path, participant)
    if len(inferred) == 1:
        return inferred
    if role == "client":
        client = canonical_client_label(participant)
        return [] if client == "unknown" else [client]

    # Server sequences must explicitly identify the client if multiple clients
    # share a file. Never mix them merely to obtain coverage.
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        client_field = next(
            (
                name
                for name in (
                    "resolved_client_id",
                    "client_capture_id",
                    "client_id",
                    "client",
                )
                if name in fields
            ),
            None,
        )
        if not client_field:
            return []
        found = set()
        for row in reader:
            client = canonical_client_label(row.get(client_field))
            if client != "unknown":
                found.add(client)
        return sorted(found)


def audit_sources(
    collected_root: Path,
    label_map: dict[tuple[str, str], dict[str, str]],
) -> dict[str, Any]:
    files = Counter()
    clients = defaultdict(set)
    experiments = defaultdict(set)
    skipped: list[dict[str, Any]] = []
    discovered: list[dict[str, Any]] = []

    # Proxy: use canonical identity resolution, not filenames.
    proxy_specs, proxy_skipped = canonical_proxy_specs(collected_root, label_map)
    skipped.extend(proxy_skipped)
    proxy_files_used = set()
    for spec in proxy_specs:
        files["proxy"] += len(spec["source_files"])
        proxy_files_used.update(spec["source_files"])
        key = (spec["run_id"], spec["client_id"])
        clients["proxy"].add(key)
        experiments["proxy"].add(spec["run_id"])
        discovered.append({
            "run_id": spec["run_id"],
            "participant": "proxy",
            "role": "proxy",
            "client_id": spec["client_id"],
            "selection_reason": spec["selection_reason"],
            "packet_count_manifest": spec["packet_count_manifest"],
            "source_files": spec["source_files"],
        })

    # files[proxy] above can include a source file twice only if a client maps
    # to multiple selected connections. Report unique file count for clarity.
    files["proxy"] = len(proxy_files_used)

    # Endpoint packet artifacts: future-compatible, but do not fabricate them.
    for run_id, participant, role, path in discover_endpoint_sequence_files(collected_root):
        try:
            identified_clients = _endpoint_clients(path, role, participant)
        except Exception as exc:
            skipped.append({
                "run_id": run_id,
                "participant": participant,
                "role": role,
                "file": str(path),
                "reason": f"read_error: {exc}",
            })
            continue
        if not identified_clients:
            skipped.append({
                "run_id": run_id,
                "participant": participant,
                "role": role,
                "file": str(path),
                "reason": "no_unambiguous_client_identity",
            })
            continue
        usable = []
        for client_id in identified_clients:
            key = (run_id, normalize_client_id(client_id))
            if key not in label_map:
                skipped.append({
                    "run_id": run_id,
                    "participant": participant,
                    "role": role,
                    "file": str(path),
                    "client_id": client_id,
                    "reason": "no_matching_ground_truth_label",
                })
                continue
            usable.append(client_id)
            clients[role].add((run_id, client_id))
            experiments[role].add(run_id)
        if usable:
            files[role] += 1
            discovered.append({
                "run_id": run_id,
                "participant": participant,
                "role": role,
                "file": str(path),
                "clients": usable,
            })

    return {
        "identity_policy": (
            "proxy traces resolved with canonical client registration/manifest "
            "mapping; endpoint traces require explicit participant/client identity"
        ),
        "sequence_files_by_role": dict(files),
        "client_traces_by_role": {k: len(v) for k, v in clients.items()},
        "experiments_by_role": {k: len(v) for k, v in experiments.items()},
        "discovered": discovered,
        "skipped": skipped,
    }


def build(args) -> dict[str, Any]:
    collected_root = Path(args.collected_root)
    y_path = Path(args.labels)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = load_labels(y_path)
    audit = audit_sources(collected_root, labels)
    audit_path = output_root / "source_coverage_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\nPacket-sequence source coverage")
    print("=" * 72)
    for role in ("client", "server", "proxy"):
        print(
            f"{role:>6}: files={audit['sequence_files_by_role'].get(role, 0)} "
            f"traces={audit['client_traces_by_role'].get(role, 0)} "
            f"experiments={audit['experiments_by_role'].get(role, 0)}"
        )
    if audit["client_traces_by_role"].get("client", 0) == 0:
        print("\n[coverage] Historical client packet sequences: unavailable")
    if audit["client_traces_by_role"].get("server", 0) == 0:
        print("[coverage] Historical server packet sequences: unavailable")
    if audit["client_traces_by_role"].get("proxy", 0):
        print("[coverage] Proxy packet traces: canonical client identity resolved")

    if args.audit_only:
        return audit

    budgets = [int(x) for x in args.packet_counts.split(",") if x.strip()]
    for k in budgets:
        if k <= 0:
            raise SystemExit("packet counts must be positive")

    handles: dict[int, Any] = {}
    writers: dict[int, csv.DictWriter] = {}
    counts = Counter()
    traces = defaultdict(set)
    experiments = defaultdict(set)

    def write_trace(
        run_id: str,
        participant: str,
        role: str,
        client_id: str,
        packets: list[PacketObservation],
        label: dict[str, str],
        source_file: str,
    ) -> None:
        if not packets:
            return
        canonical_client = canonical_client_label(client_id)
        for k in budgets:
            for block_index, start, end, features in iter_packet_budget_samples(packets, k):
                row = {
                    "sample_id": f"{run_id}:{canonical_client}:{role}:k{k}:{block_index}",
                    "experiment_id": run_id,
                    "client_id": canonical_client,
                    "source_role": role,
                    "source_participant": participant,
                    "source_file": source_file,
                    "packet_budget": k,
                    "block_index": block_index,
                    "packet_start_index": start,
                    "packet_end_index": end,
                    **label,
                    **features,
                }
                if k not in handles:
                    out = output_root / f"packet_{k}.csv"
                    handle = out.open("w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                    writer.writeheader()
                    handles[k] = handle
                    writers[k] = writer
                writers[k].writerow(row)
                counts[(k, role)] += 1
                traces[(k, role)].add((run_id, canonical_client))
                experiments[(k, role)].add(run_id)

    try:
        # Canonical proxy traces.
        proxy_specs, _ = canonical_proxy_specs(collected_root, labels)
        for idx, spec in enumerate(proxy_specs, start=1):
            print(
                f"[proxy {idx}/{len(proxy_specs)}] "
                f"{spec['run_id']} {spec['client_id']} "
                f"manifest_packets={spec['packet_count_manifest']:,}"
            )
            packets = load_canonical_proxy_packets(spec)
            write_trace(
                spec["run_id"],
                "proxy",
                "proxy",
                spec["client_id"],
                packets,
                spec["label"],
                ";".join(spec["source_files"]),
            )

        # Endpoint packet traces when they exist in future/new runs.
        for run_id, participant, role, path in discover_endpoint_sequence_files(collected_root):
            try:
                grouped = read_sequence_groups(path, role, participant)
            except Exception:
                continue
            for client_id, packets in grouped.items():
                label = labels.get((run_id, normalize_client_id(client_id)))
                if label is None:
                    continue
                write_trace(
                    run_id,
                    participant,
                    role,
                    client_id,
                    packets,
                    label,
                    str(path),
                )
    finally:
        for handle in handles.values():
            handle.close()

    inventory_path = output_root / "packet_dataset_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "packet_budget",
                "source_role",
                "samples",
                "client_traces",
                "experiments",
            ],
        )
        writer.writeheader()
        for k in budgets:
            for role in ("client", "server", "proxy"):
                writer.writerow({
                    "packet_budget": k,
                    "source_role": role,
                    "samples": counts[(k, role)],
                    "client_traces": len(traces[(k, role)]),
                    "experiments": len(experiments[(k, role)]),
                })

    summary = {
        "packet_counts": budgets,
        "source_coverage_audit": str(audit_path),
        "inventory": str(inventory_path),
        "samples": {
            f"k{k}:{role}": counts[(k, role)]
            for k in budgets
            for role in ("client", "server", "proxy")
        },
        "method": "non_overlapping_fixed_packet_blocks; k=1 uses every packet",
        "proxy_identity_resolution": "canonical registration+manifest mapping",
        "direction_semantics": "up=client_to_server; down=server_to_client",
        "payload_policy": "encrypted payload content is not used",
    }
    (output_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected-root", default="collected_experiments")
    parser.add_argument(
        "--labels", default="fingerprinting_dataset/fingerprinting_Y_ground_truth.csv"
    )
    parser.add_argument(
        "--output", default="fingerprinting_dataset/packet_cross_vantage"
    )
    parser.add_argument(
        "--packet-counts", default=",".join(map(str, PACKET_COUNTS))
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = build(args)
    print("\nDone.")
    if isinstance(result, dict) and "inventory" in result:
        print("Inventory:", result["inventory"])


if __name__ == "__main__":
    main()
