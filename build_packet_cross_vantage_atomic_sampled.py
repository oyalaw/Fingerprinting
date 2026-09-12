from __future__ import annotations

"""Atomic, trace-balanced packet-budget dataset builder.

This script reuses the discovery/identity logic from the existing
build_packet_cross_vantage_dataset.py, but changes the materialization policy:

- it never writes all hundreds of millions of packet observations;
- it deterministically selects at most N observations per client trace and K;
- it writes into a staging directory;
- it validates actual written coverage against the discovered source traces;
- only after successful validation does it atomically promote the new dataset;
- an interrupted build therefore cannot truncate the last good dataset.

The selected observation semantics are exactly the original semantics:
K=1 -> one arriving packet plus the IAT to its immediate predecessor.
K>1 -> non-overlapping exact-K blocks from the original trace.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_packet_cross_vantage_dataset as legacy

from ai_fingerprint.packet_fingerprinting import (
    packet_block_features,
    single_packet_features,
)


def canonical_trace_key(
    run_id: str,
    client_id: str,
    role: str,
) -> tuple[str, str, str]:
    return (
        str(run_id),
        str(legacy.canonical_client_label(client_id)),
        str(role),
    )


def stable_seed(
    global_seed: int,
    packet_budget: int,
    trace_key: tuple[str, str, str],
) -> int:
    material = (
        f"{int(global_seed)}|{int(packet_budget)}|"
        + "|".join(trace_key)
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def stratified_positions(
    n_units: int,
    cap: int,
    seed: int,
) -> list[int]:
    """Deterministically cover the whole trace with <= cap unit indices."""
    n_units = int(n_units)
    cap = int(cap)
    if n_units <= 0:
        return []
    if cap <= 0 or n_units <= cap:
        return list(range(n_units))

    # Pure-Python deterministic one-per-stratum selection. This avoids a
    # dependency on NumPy in the builder and spans the whole trace.
    import random
    rng = random.Random(int(seed))
    positions: list[int] = []

    for i in range(cap):
        lo = (i * n_units) // cap
        hi = ((i + 1) * n_units) // cap
        if hi <= lo:
            hi = lo + 1
        positions.append(rng.randrange(lo, min(hi, n_units)))

    # Strata are disjoint for n_units >= cap, so uniqueness is expected.
    return positions


def selected_units_for_budget(
    packets,
    packet_budget: int,
    max_samples_per_trace: int,
    seed: int,
):
    """Yield (block_index, start, end_inclusive, features)."""
    k = int(packet_budget)
    if k <= 0:
        raise ValueError("packet_budget must be positive")

    n_packets = len(packets)
    n_units = n_packets if k == 1 else n_packets // k
    positions = stratified_positions(
        n_units,
        max_samples_per_trace,
        seed,
    )

    if k == 1:
        for index in positions:
            packet = packets[index]
            previous = (
                None
                if index == 0
                else packets[index - 1].timestamp
            )
            yield (
                index,
                index,
                index,
                single_packet_features(packet, previous),
            )
        return

    for block_index in positions:
        start = block_index * k
        end_exclusive = start + k
        block = packets[start:end_exclusive]
        # Exact-K semantics are preserved.
        if len(block) != k:
            continue
        yield (
            block_index,
            start,
            end_exclusive - 1,
            packet_block_features(block),
        )


class AtomicPacketWriter:
    def __init__(
        self,
        staging_root: Path,
        budgets: list[int],
    ):
        self.staging_root = staging_root
        self.budgets = budgets
        self.handles: dict[int, Any] = {}
        self.writers: dict[int, csv.DictWriter] = {}
        self.counts = Counter()
        self.traces = defaultdict(set)
        self.experiments = defaultdict(set)
        self.expected_traces = defaultdict(set)
        self.expected_experiments = defaultdict(set)
        self.available_units = Counter()
        self.trace_audit: list[dict[str, Any]] = []

    def _writer_for(
        self,
        k: int,
        row: dict[str, Any],
    ) -> csv.DictWriter:
        if k in self.writers:
            return self.writers[k]

        out = self.staging_root / f"packet_{k}.csv"
        handle = out.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row.keys()),
        )
        writer.writeheader()
        self.handles[k] = handle
        self.writers[k] = writer
        return writer

    def write_trace(
        self,
        *,
        run_id: str,
        participant: str,
        role: str,
        client_id: str,
        packets,
        label: dict[str, str],
        source_file: str,
        max_samples_per_trace: int,
        sampling_seed: int,
    ) -> None:
        if not packets:
            return

        canonical_client = legacy.canonical_client_label(
            client_id
        )
        trace_key = canonical_trace_key(
            run_id,
            canonical_client,
            role,
        )

        for k in self.budgets:
            n_units = (
                len(packets)
                if k == 1
                else len(packets) // k
            )
            if n_units <= 0:
                continue

            self.expected_traces[(k, role)].add(
                (run_id, canonical_client)
            )
            self.expected_experiments[(k, role)].add(
                run_id
            )
            self.available_units[(k, role)] += int(n_units)

            seed = stable_seed(
                sampling_seed,
                k,
                trace_key,
            )
            selected_count = 0

            for (
                block_index,
                start,
                end,
                features,
            ) in selected_units_for_budget(
                packets,
                k,
                max_samples_per_trace,
                seed,
            ):
                row = {
                    "sample_id": (
                        f"{run_id}:{canonical_client}:"
                        f"{role}:k{k}:{block_index}"
                    ),
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

                self._writer_for(k, row).writerow(row)
                self.counts[(k, role)] += 1
                self.traces[(k, role)].add(
                    (run_id, canonical_client)
                )
                self.experiments[(k, role)].add(run_id)
                selected_count += 1

            self.trace_audit.append({
                "packet_budget": int(k),
                "experiment_id": run_id,
                "client_id": canonical_client,
                "source_role": role,
                "source_participant": participant,
                "source_file": source_file,
                "packet_count": int(len(packets)),
                "available_units": int(n_units),
                "selected_units": int(selected_count),
                "max_samples_per_trace": int(
                    max_samples_per_trace
                ),
                "sampling_seed": int(seed),
                "sampling_method": (
                    "deterministic_stratified_across_trace"
                ),
            })

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()
        self.writers.clear()

    def validate(self) -> dict[str, Any]:
        problems: list[str] = []
        coverage = {}

        for k in self.budgets:
            for role in ("client", "server", "proxy"):
                expected_traces = self.expected_traces[
                    (k, role)
                ]
                actual_traces = self.traces[(k, role)]
                expected_exps = self.expected_experiments[
                    (k, role)
                ]
                actual_exps = self.experiments[(k, role)]

                missing_traces = sorted(
                    expected_traces - actual_traces
                )
                extra_traces = sorted(
                    actual_traces - expected_traces
                )
                missing_exps = sorted(
                    expected_exps - actual_exps
                )

                key = f"k{k}:{role}"
                coverage[key] = {
                    "available_units": int(
                        self.available_units[(k, role)]
                    ),
                    "written_samples": int(
                        self.counts[(k, role)]
                    ),
                    "expected_client_traces": int(
                        len(expected_traces)
                    ),
                    "written_client_traces": int(
                        len(actual_traces)
                    ),
                    "expected_experiments": int(
                        len(expected_exps)
                    ),
                    "written_experiments": int(
                        len(actual_exps)
                    ),
                    "missing_traces": [
                        list(x)
                        for x in missing_traces
                    ],
                    "extra_traces": [
                        list(x)
                        for x in extra_traces
                    ],
                    "missing_experiments": missing_exps,
                }

                if missing_traces:
                    problems.append(
                        f"{key}: missing "
                        f"{len(missing_traces)} expected traces"
                    )
                if extra_traces:
                    problems.append(
                        f"{key}: contains "
                        f"{len(extra_traces)} unexpected traces"
                    )
                if missing_exps:
                    problems.append(
                        f"{key}: missing "
                        f"{len(missing_exps)} expected experiments"
                    )

        return {
            "status": "valid" if not problems else "invalid",
            "problems": problems,
            "coverage": coverage,
        }


def write_inventory(
    staging_root: Path,
    writer_state: AtomicPacketWriter,
) -> Path:
    path = staging_root / "packet_dataset_inventory.csv"
    fields = [
        "packet_budget",
        "source_role",
        "samples",
        "available_samples",
        "client_traces",
        "experiments",
        "expected_client_traces",
        "expected_experiments",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for k in writer_state.budgets:
            for role in ("client", "server", "proxy"):
                writer.writerow({
                    "packet_budget": k,
                    "source_role": role,
                    # Compatibility: samples means actual rows written.
                    "samples": int(
                        writer_state.counts[(k, role)]
                    ),
                    "available_samples": int(
                        writer_state.available_units[(k, role)]
                    ),
                    "client_traces": int(
                        len(writer_state.traces[(k, role)])
                    ),
                    "experiments": int(
                        len(
                            writer_state.experiments[
                                (k, role)
                            ]
                        )
                    ),
                    "expected_client_traces": int(
                        len(
                            writer_state.expected_traces[
                                (k, role)
                            ]
                        )
                    ),
                    "expected_experiments": int(
                        len(
                            writer_state.expected_experiments[
                                (k, role)
                            ]
                        )
                    ),
                })

    return path


def write_trace_audit(
    staging_root: Path,
    rows: list[dict[str, Any]],
) -> Path:
    path = staging_root / "packet_trace_sampling_audit.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fields = list(rows[0].keys())
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def promote_atomically(
    staging_root: Path,
    output_root: Path,
    *,
    keep_backup: bool,
) -> Path | None:
    """Promote complete staging directory only after validation succeeds."""
    backup = None

    if output_root.exists():
        stamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        backup = output_root.with_name(
            output_root.name + f".backup_{stamp}"
        )
        output_root.rename(backup)

    try:
        staging_root.rename(output_root)
    except Exception:
        # Restore the previous dataset if promotion itself fails.
        if backup is not None and backup.exists():
            if output_root.exists():
                shutil.rmtree(output_root)
            backup.rename(output_root)
        raise

    if (
        backup is not None
        and backup.exists()
        and not keep_backup
    ):
        shutil.rmtree(backup)
        backup = None

    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collected-root",
        default="collected_experiments",
    )
    parser.add_argument(
        "--labels",
        default=(
            "fingerprinting_dataset/"
            "fingerprinting_Y_ground_truth.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "fingerprinting_dataset/"
            "packet_cross_vantage"
        ),
    )
    parser.add_argument(
        "--packet-counts",
        default="1,5,10,25,50,100,250,500",
    )
    parser.add_argument(
        "--max-samples-per-trace",
        type=int,
        default=5000,
        help=(
            "Maximum selected observations per "
            "experiment/client/source trace for each K."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--keep-backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the previously existing dataset directory "
            "after successful promotion. Default: yes."
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
    )
    args = parser.parse_args()

    if args.max_samples_per_trace <= 0:
        raise SystemExit(
            "--max-samples-per-trace must be positive"
        )

    budgets = [
        int(value)
        for value in args.packet_counts.split(",")
        if value.strip()
    ]
    if not budgets or any(k <= 0 for k in budgets):
        raise SystemExit(
            "packet counts must be positive"
        )

    collected_root = Path(args.collected_root)
    labels_path = Path(args.labels)
    output_root = Path(args.output)

    labels = legacy.load_labels(labels_path)
    source_audit = legacy.audit_sources(
        collected_root,
        labels,
    )

    print("\nPacket-sequence source coverage")
    print("=" * 72)
    for role in ("client", "server", "proxy"):
        print(
            f"{role:>6}: "
            f"files={source_audit['sequence_files_by_role'].get(role, 0)} "
            f"traces={source_audit['client_traces_by_role'].get(role, 0)} "
            f"experiments={source_audit['experiments_by_role'].get(role, 0)}"
        )

    if args.audit_only:
        print(
            json.dumps(
                source_audit,
                indent=2,
            )
        )
        return

    staging_root = output_root.with_name(
        output_root.name
        + f".building_{os.getpid()}"
    )
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    (staging_root / "source_coverage_audit.json").write_text(
        json.dumps(
            source_audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    state = AtomicPacketWriter(
        staging_root,
        budgets,
    )

    try:
        proxy_specs, _ = legacy.canonical_proxy_specs(
            collected_root,
            labels,
        )

        print(
            f"\n[build] Canonical proxy traces: "
            f"{len(proxy_specs)}"
        )

        for idx, spec in enumerate(
            proxy_specs,
            start=1,
        ):
            print(
                f"[proxy {idx}/{len(proxy_specs)}] "
                f"{spec['run_id']} "
                f"{spec['client_id']} "
                f"manifest_packets="
                f"{spec['packet_count_manifest']:,}"
            )

            packets = legacy.load_canonical_proxy_packets(
                spec
            )

            print(
                f"  loaded_packets={len(packets):,}"
            )

            state.write_trace(
                run_id=spec["run_id"],
                participant="proxy",
                role="proxy",
                client_id=spec["client_id"],
                packets=packets,
                label=spec["label"],
                source_file=";".join(
                    spec["source_files"]
                ),
                max_samples_per_trace=(
                    args.max_samples_per_trace
                ),
                sampling_seed=args.sampling_seed,
            )

            # Explicitly drop the potentially large trace before loading
            # the next client.
            del packets

        # Future endpoint packet traces, if present.
        for (
            run_id,
            participant,
            role,
            path,
        ) in legacy.discover_endpoint_sequence_files(
            collected_root
        ):
            try:
                grouped = legacy.read_sequence_groups(
                    path,
                    role,
                    participant,
                )
            except Exception as exc:
                print(
                    f"[endpoint skip] {path}: {exc}"
                )
                continue

            for client_id, packets in grouped.items():
                label = labels.get(
                    (
                        run_id,
                        legacy.normalize_client_id(
                            client_id
                        ),
                    )
                )
                if label is None:
                    continue

                state.write_trace(
                    run_id=run_id,
                    participant=participant,
                    role=role,
                    client_id=client_id,
                    packets=packets,
                    label=label,
                    source_file=str(path),
                    max_samples_per_trace=(
                        args.max_samples_per_trace
                    ),
                    sampling_seed=args.sampling_seed,
                )

        state.close()

        validation = state.validate()
        (
            staging_root
            / "dataset_validation.json"
        ).write_text(
            json.dumps(
                validation,
                indent=2,
            ),
            encoding="utf-8",
        )

        write_inventory(
            staging_root,
            state,
        )
        write_trace_audit(
            staging_root,
            state.trace_audit,
        )

        summary = {
            "packet_counts": budgets,
            "source_coverage_audit": (
                "source_coverage_audit.json"
            ),
            "inventory": (
                "packet_dataset_inventory.csv"
            ),
            "trace_sampling_audit": (
                "packet_trace_sampling_audit.csv"
            ),
            "validation": (
                "dataset_validation.json"
            ),
            "sampling_policy": (
                "deterministic_stratified_per_"
                "experiment_client_source_trace"
            ),
            "sampling_seed": int(
                args.sampling_seed
            ),
            "max_samples_per_trace": int(
                args.max_samples_per_trace
            ),
            "method": (
                "K=1 exact packet observation with immediate "
                "predecessor IAT; K>1 non-overlapping exact-K "
                "blocks; only selected blocks are materialized"
            ),
            "proxy_identity_resolution": (
                "canonical registration+manifest mapping"
            ),
            "direction_semantics": (
                "up=client_to_server; "
                "down=server_to_client"
            ),
            "payload_policy": (
                "encrypted payload content is not used"
            ),
            "available_samples": {
                f"k{k}:{role}": int(
                    state.available_units[(k, role)]
                )
                for k in budgets
                for role in (
                    "client",
                    "server",
                    "proxy",
                )
            },
            "written_samples": {
                f"k{k}:{role}": int(
                    state.counts[(k, role)]
                )
                for k in budgets
                for role in (
                    "client",
                    "server",
                    "proxy",
                )
            },
        }

        (
            staging_root / "build_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        if validation["status"] != "valid":
            raise RuntimeError(
                "Dataset validation failed; staging dataset "
                f"was NOT promoted. Problems: "
                f"{validation['problems']}"
            )

        # Strong current-campaign proxy gate: discovered source coverage and
        # written K=1 coverage must agree exactly.
        source_proxy_traces = int(
            source_audit[
                "client_traces_by_role"
            ].get("proxy", 0)
        )
        source_proxy_exps = int(
            source_audit[
                "experiments_by_role"
            ].get("proxy", 0)
        )
        written_k1_traces = len(
            state.traces[(1, "proxy")]
        ) if 1 in budgets else None
        written_k1_exps = len(
            state.experiments[(1, "proxy")]
        ) if 1 in budgets else None

        if 1 in budgets and (
            written_k1_traces != source_proxy_traces
            or written_k1_exps != source_proxy_exps
        ):
            raise RuntimeError(
                "K=1 proxy coverage does not match discovered "
                "canonical source coverage: "
                f"source={source_proxy_traces} traces/"
                f"{source_proxy_exps} experiments, "
                f"written={written_k1_traces} traces/"
                f"{written_k1_exps} experiments. "
                "Staging dataset was NOT promoted."
            )

        backup = promote_atomically(
            staging_root,
            output_root,
            keep_backup=args.keep_backup,
        )

        print("\nBUILD COMPLETE")
        print("=" * 72)
        print(f"Promoted dataset: {output_root}")
        if backup is not None:
            print(f"Previous dataset backup: {backup}")

        inventory = output_root / "packet_dataset_inventory.csv"
        print(f"Inventory: {inventory}")

    except BaseException:
        # Ensure open handles are closed. Do NOT promote partial outputs.
        state.close()
        print(
            "\n[atomic-build] Build did not complete. "
            f"Existing dataset remains untouched. "
            f"Partial staging directory: {staging_root}"
        )
        raise


if __name__ == "__main__":
    main()
