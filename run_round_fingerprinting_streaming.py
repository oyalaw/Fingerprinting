from __future__ import annotations

import argparse
import csv
import hashlib
import gc
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DETECTOR_BUILD_VERSION = (
    "round-adaptive-major-v2-"
    "robust-clock-strict-iou-v4-tol025"
)

import build_packet_cross_vantage_dataset as legacy

from ai_fingerprint.fingerprinting_dataset import (
    _read_ground_truth_indices,
)
from ai_fingerprint.result_collection import (
    validate_collected_root,
)
from ai_fingerprint.round_fingerprinting import (
    RoundFingerprintingError,
    RoundInferenceConfig,
    build_round_dataset,
    infer_round_boundaries,
    read_ground_truth_rounds,
    validate_inferred_rounds,
)
from prepare_fingerprinting_dataset import discover_inputs


def now_tag() -> str:
    return datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_"}
        else "_"
        for ch in text
    )


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def load_canonical_packet_records(
    spec: dict[str, Any],
):
    """
    Load ONE logical client trace using the exact canonical
    source-selection policy used by the successful packet-K
    pipeline.

    Unlike legacy.load_canonical_proxy_packets(), this keeps
    PacketRecord objects because round inference requires
    timestamp_epoch and the original directional metadata.
    """
    wanted = {
        tuple(value)
        for value in spec["connections"]
    }

    connection_numbers = {
        connection: number
        for number, connection in enumerate(
            sorted(wanted),
            start=1,
        )
    }

    packets = []

    for entry in spec["entries"]:
        packets.extend(
            legacy._read_safe_sequence(
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

    return packets


def read_csv_dicts(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def union_fields(
    rows: list[dict[str, Any]],
    preferred: list[str] | None = None,
) -> list[str]:
    seen = set()
    result = []

    for field in preferred or []:
        if field not in seen:
            seen.add(field)
            result.append(field)

    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                result.append(field)

    return result


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })


def inference_config_signature(
    cfg: RoundInferenceConfig,
) -> str:
    payload = {
        "detector_build_version":
            DETECTOR_BUILD_VERSION,
        "config": {
            key: value
            for key, value
            in vars(cfg).items()
        },
    }

    material = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        material
    ).hexdigest()


def part_complete(
    part_dir: Path,
    expected_signature: str,
) -> bool:
    status = (
        part_dir
        / "streaming_part_status.json"
    )

    if not status.exists():
        return False

    try:
        payload = json.loads(
            status.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return False

    return bool(
        payload.get("complete")
        and payload.get(
            "config_signature"
        )
        == expected_signature
    )

def build_one_trace(
    *,
    spec: dict[str, Any],
    ground_truth_rounds,
    cfg: RoundInferenceConfig,
    part_dir: Path,
    config_signature: str,
) -> dict[str, Any]:
    run_id = str(
        spec["run_id"]
    )

    client_id = str(
        spec["client_id"]
    )

    capture_id = (
        f"canonical_{client_id}"
    )

    if part_dir.exists():
        shutil.rmtree(part_dir)

    part_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"  loading canonical packets...",
        flush=True,
    )

    packets = load_canonical_packet_records(
        spec
    )

    packet_count = len(packets)

    print(
        f"  loaded_packets={packet_count:,}",
        flush=True,
    )

    if packet_count < 20:
        raise RuntimeError(
            f"{run_id}/{client_id}: only "
            f"{packet_count} canonical packets"
        )

    trace_client_labels = {
        (
            run_id,
            client_id,
        ): spec["label"]
    }

    try:
        result = build_round_dataset(
            client_packets={
                (
                    run_id,
                    capture_id,
                    client_id,
                ): packets
            },
            client_labels=(
                trace_client_labels
            ),
            ground_truth_rounds=(
                ground_truth_rounds
            ),
            output_dir=part_dir,
            inference_config=cfg,
        )

        row_count = count_csv_rows(
            Path(result["x_csv"])
        )

        status = {
            "complete": True,
            "status": "created",
            "experiment_id": run_id,
            "client_id": client_id,
            "capture_id": capture_id,
            "packet_count": packet_count,
            "round_sample_count":
                row_count,
            "source_files":
                spec["source_files"],
            "selection_reason":
                spec.get(
                    "selection_reason",
                    "",
                ),
        }

    except RoundFingerprintingError as exc:
        message = str(exc)

        if (
            "No round-level samples were created"
            not in message
        ):
            raise

        # A trace with zero usable inferred rounds is a
        # legitimate scientific outcome. Preserve its
        # inference and validation diagnostics rather
        # than silently discarding the trace.
        inferred, diagnostics = (
            infer_round_boundaries(
                packets,
                cfg,
            )
        )

        truth = list(
            ground_truth_rounds.get(
                (
                    run_id,
                    client_id,
                ),
                [],
            )
        )

        validation = validate_inferred_rounds(
            inferred,
            truth,
        )

        key = (
            f"{run_id}::"
            f"{client_id}::"
            f"{capture_id}"
        )

        (
            part_dir
            / "round_inference_diagnostics.json"
        ).write_text(
            json.dumps(
                {
                    key: diagnostics
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        (
            part_dir
            / "round_boundary_validation.json"
        ).write_text(
            json.dumps(
                {
                    key: validation
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        status = {
            "complete": True,
            "status":
                "zero_usable_round_samples",
            "experiment_id": run_id,
            "client_id": client_id,
            "capture_id": capture_id,
            "packet_count": packet_count,
            "round_sample_count": 0,
            "inferred_round_count":
                len(inferred),
            "source_files":
                spec["source_files"],
            "selection_reason":
                spec.get(
                    "selection_reason",
                    "",
                ),
        }

    status[
        "detector_build_version"
    ] = DETECTOR_BUILD_VERSION

    status[
        "config_signature"
    ] = config_signature

    (
        part_dir
        / "streaming_part_status.json"
    ).write_text(
        json.dumps(
            status,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    del packets
    gc.collect()

    return status


def merge_parts(
    *,
    specs: list[dict[str, Any]],
    work_root: Path,
    output_dir: Path,
    cfg: RoundInferenceConfig,
) -> dict[str, Any]:

    all_x = []
    all_y = []
    all_boundaries = []

    all_validation = {}
    all_diagnostics = {}

    predictor_columns = None
    label_columns = []

    statuses = []

    for spec in specs:
        run_id = str(
            spec["run_id"]
        )
        client_id = str(
            spec["client_id"]
        )

        part_dir = (
            work_root
            / "parts"
            / (
                safe_name(run_id)
                + "__"
                + safe_name(client_id)
            )
        )

        status_path = (
            part_dir
            / "streaming_part_status.json"
        )

        if not status_path.exists():
            raise RuntimeError(
                f"Missing completed part: "
                f"{run_id}/{client_id}"
            )

        status = json.loads(
            status_path.read_text(
                encoding="utf-8"
            )
        )

        if not status.get("complete"):
            raise RuntimeError(
                f"Incomplete part: "
                f"{run_id}/{client_id}"
            )

        statuses.append(status)

        x_path = (
            part_dir
            / "round_X_proxy.csv"
        )
        y_path = (
            part_dir
            / "round_Y_ground_truth.csv"
        )
        b_path = (
            part_dir
            / "round_boundaries.csv"
        )
        s_path = (
            part_dir
            / "round_schema.json"
        )
        v_path = (
            part_dir
            / "round_boundary_validation.json"
        )
        d_path = (
            part_dir
            / "round_inference_diagnostics.json"
        )

        x_rows = read_csv_dicts(
            x_path
        )
        y_rows = read_csv_dicts(
            y_path
        )

        if len(x_rows) != len(y_rows):
            raise RuntimeError(
                f"{run_id}/{client_id}: "
                f"X/Y mismatch "
                f"{len(x_rows)} != "
                f"{len(y_rows)}"
            )

        all_x.extend(x_rows)
        all_y.extend(y_rows)

        all_boundaries.extend(
            read_csv_dicts(b_path)
        )

        if s_path.exists():
            schema = json.loads(
                s_path.read_text(
                    encoding="utf-8"
                )
            )

            current = list(
                schema.get(
                    "predictor_columns",
                    [],
                )
            )

            if predictor_columns is None:
                predictor_columns = current
            elif (
                current
                and current
                != predictor_columns
            ):
                raise RuntimeError(
                    "Predictor schema differs "
                    f"for {run_id}/{client_id}"
                )

            for field in schema.get(
                "label_columns",
                [],
            ):
                if field not in label_columns:
                    label_columns.append(
                        field
                    )

        if v_path.exists():
            payload = json.loads(
                v_path.read_text(
                    encoding="utf-8"
                )
            )

            overlap = (
                set(all_validation)
                & set(payload)
            )

            if overlap:
                raise RuntimeError(
                    "Duplicate validation keys: "
                    f"{sorted(overlap)}"
                )

            all_validation.update(
                payload
            )

        if d_path.exists():
            payload = json.loads(
                d_path.read_text(
                    encoding="utf-8"
                )
            )

            overlap = (
                set(all_diagnostics)
                & set(payload)
            )

            if overlap:
                raise RuntimeError(
                    "Duplicate diagnostic keys: "
                    f"{sorted(overlap)}"
                )

            all_diagnostics.update(
                payload
            )

    if not all_x:
        raise RuntimeError(
            "No round samples were produced "
            "by any logical client trace."
        )

    if predictor_columns is None:
        raise RuntimeError(
            "No round predictor schema found."
        )

    # Part builds start row_id at 1 independently.
    # Reassign globally unique row IDs while preserving
    # exact X/Y row alignment.
    for row_id, (
        x_row,
        y_row,
    ) in enumerate(
        zip(all_x, all_y),
        start=1,
    ):
        x_row["row_id"] = row_id
        y_row["row_id"] = row_id

    staging = output_dir.with_name(
        output_dir.name
        + ".staging."
        + now_tag()
    )

    if staging.exists():
        shutil.rmtree(staging)

    staging.mkdir(
        parents=True,
        exist_ok=False,
    )

    x_fields = [
        "row_id",
        "experiment_id",
        "client_capture_id",
        "row_type",
        *predictor_columns,
    ]

    y_fields = union_fields(
        all_y,
        preferred=[
            "row_id",
            "experiment_id",
            "client_capture_id",
            "resolved_client_id",
            "inferred_round_index",
            "round_boundary_confidence",
            *label_columns,
        ],
    )

    boundary_fields = union_fields(
        all_boundaries
    )

    write_csv(
        staging / "round_X_proxy.csv",
        all_x,
        x_fields,
    )

    write_csv(
        staging
        / "round_Y_ground_truth.csv",
        all_y,
        y_fields,
    )

    if all_boundaries:
        write_csv(
            staging
            / "round_boundaries.csv",
            all_boundaries,
            boundary_fields,
        )

    (
        staging
        / "round_boundary_validation.json"
    ).write_text(
        json.dumps(
            all_validation,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (
        staging
        / "round_inference_diagnostics.json"
    ).write_text(
        json.dumps(
            all_diagnostics,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    experiment_ids = {
        str(row.get(
            "experiment_id",
            "",
        ))
        for row in all_y
        if row.get(
            "experiment_id",
            ""
        )
    }

    trace_ids = {
        (
            str(row.get(
                "experiment_id",
                "",
            )),
            str(row.get(
                "resolved_client_id",
                "",
            )),
        )
        for row in all_y
        if row.get(
            "experiment_id",
            ""
        )
    }

    schema = {
        "schema_version":
            "round-streaming-v1",
        "representation":
            "one inferred FL round for one logical client",
        "segmentation_source":
            "proxy_observable_traffic_only",
        "canonical_source_policy":
            (
                "same canonical_proxy_specs and "
                "manifest-backed selected FL "
                "connections used by packet-K "
                "evaluation"
            ),
        "logical_clients_never_merged":
            True,
        "same_client_reconnects_stitched":
            True,
        "ground_truth_round_boundaries_used_as_predictors":
            False,
        "true_round_index_used_as_predictor":
            False,
        "inferred_round_index_used_as_predictor":
            False,
        "round_boundary_confidence_used_as_predictor":
            False,
        "predictor_columns":
            predictor_columns,
        "metadata_columns": [
            "row_id",
            "experiment_id",
            "client_capture_id",
            "row_type",
        ],
        "label_columns":
            label_columns,
        "row_count":
            len(all_x),
        "experiment_count":
            len(experiment_ids),
        "client_trace_count":
            len(trace_ids),
        "source_trace_count":
            len(specs),
        "zero_sample_trace_count":
            sum(
                1
                for value in statuses
                if int(
                    value.get(
                        "round_sample_count",
                        0,
                    )
                ) == 0
            ),
        "inference_config":
            {
                key: value
                for key, value
                in vars(cfg).items()
            },
        "evaluation_rule":
            (
                "All inferred rounds from one "
                "experiment remain in the same "
                "train/test fold. Random round "
                "splitting within an experiment "
                "is forbidden."
            ),
        "streaming_build":
            True,
        "per_trace_checkpointing":
            True,
    }

    (
        staging
        / "round_schema.json"
    ).write_text(
        json.dumps(
            schema,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    build_audit = {
        "canonical_trace_count":
            len(specs),
        "canonical_experiment_count":
            len({
                spec["run_id"]
                for spec in specs
            }),
        "statuses":
            statuses,
    }

    (
        staging
        / "streaming_build_audit.json"
    ).write_text(
        json.dumps(
            build_audit,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup = None

    if output_dir.exists():
        backup = output_dir.with_name(
            output_dir.name
            + ".backup."
            + now_tag()
        )

        output_dir.rename(
            backup
        )

    os.replace(
        staging,
        output_dir,
    )

    return {
        "output_dir":
            str(output_dir),
        "backup":
            str(backup)
            if backup
            else None,
        "row_count":
            len(all_x),
        "experiment_count":
            len(experiment_ids),
        "client_trace_count":
            len(trace_ids),
        "source_trace_count":
            len(specs),
        "zero_sample_trace_count":
            schema[
                "zero_sample_trace_count"
            ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Memory-safe, resumable round "
            "fingerprinting dataset builder "
            "using canonical manifest-backed "
            "proxy traces."
        )
    )

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
        help=(
            "Authoritative client label CSV. "
            "Uses the same label source as the "
            "packet-K experiment."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "fingerprinting_dataset/"
            "round_fingerprinting"
        ),
    )

    parser.add_argument(
        "--work-dir",
        default=(
            "fingerprinting_dataset/"
            "round_fingerprinting_work"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse completed per-client "
            "checkpoints."
        ),
    )

    parser.add_argument(
        "--bin-sec",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--direction-dominance",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--min-major-transfer-bytes",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--bridge-gap-sec",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--max-round-sec",
        type=float,
        default=3600.0,
    )

    parser.add_argument(
        "--min-round-packets",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    collected_root = Path(
        args.collected_root
    ).resolve()

    output_dir = Path(
        args.output_dir
    )

    work_root = Path(
        args.work_dir
    )

    validation = validate_collected_root(
        collected_root
    )

    valid_run_ids = list(
        validation.get(
            "valid_run_ids"
        )
        or []
    )

    if not valid_run_ids:
        raise SystemExit(
            "No VALID centrally collected "
            "runs are available."
        )

    print(
        f"VALID runs: "
        f"{len(valid_run_ids)}",
        flush=True,
    )

    (
        proxy_features,
        ground_truth,
        client_map,
        diagnostics,
    ) = discover_inputs(
        collected_root,
        allowed_experiment_ids=set(
            valid_run_ids
        ),
    )

    if not ground_truth:
        raise SystemExit(
            "No matching ground-truth "
            "JSONL files found."
        )

    # Use exactly the same authoritative client-label
    # representation as the successful packet-K builder.
    # Endpoint JSONL records remain separate and are used
    # only for validating inferred round boundaries.
    labels_path = Path(
        args.labels
    )

    if not labels_path.exists():
        raise SystemExit(
            f"Missing label dataset: "
            f"{labels_path}"
        )

    labels = legacy.load_labels(
        labels_path
    )

    valid_set = set(
        valid_run_ids
    )

    labels = {
        key: value
        for key, value in labels.items()
        if key[0] in valid_set
    }

    print(
        f"Authoritative client labels: "
        f"{len(labels)}",
        flush=True,
    )

    print(
        f"Label experiments: "
        f"{len({key[0] for key in labels})}",
        flush=True,
    )

    ground_truth_rounds = (
        read_ground_truth_rounds(
            ground_truth
        )
    )

    # Same canonical identity, manifest, reconnect, and
    # connection-selection policy as packet-K evaluation.
    specs, skipped = (
        legacy.canonical_proxy_specs(
            collected_root,
            labels,
        )
    )

    print(
        "\nCanonical proxy source coverage",
        flush=True,
    )
    print(
        f"  traces      : {len(specs)}",
        flush=True,
    )

    run_counts = Counter(
        str(spec["run_id"])
        for spec in specs
    )

    print(
        f"  experiments : "
        f"{len(run_counts)}",
        flush=True,
    )

    bad_runs = {
        run_id: count
        for run_id, count
        in run_counts.items()
        if count != 3
    }

    if skipped:
        print(
            f"  resolver skipped entries: "
            f"{len(skipped)}",
            flush=True,
        )

        for item in skipped[:20]:
            print(
                "   ",
                item,
                flush=True,
            )

    if (
        len(valid_run_ids) == 32
        and (
            len(specs) != 96
            or len(run_counts) != 32
            or bad_runs
        )
    ):
        raise SystemExit(
            "Canonical source coverage is not "
            "32 experiments x 3 logical clients. "
            f"traces={len(specs)}, "
            f"experiments={len(run_counts)}, "
            f"bad_runs={bad_runs}"
        )

    cfg = RoundInferenceConfig(
        bin_sec=args.bin_sec,
        direction_dominance=(
            args.direction_dominance
        ),
        min_major_transfer_bytes=(
            args.min_major_transfer_bytes
        ),
        bridge_gap_sec=(
            args.bridge_gap_sec
        ),
        max_round_sec=(
            args.max_round_sec
        ),
        min_round_packets=(
            args.min_round_packets
        ),
    )

    config_signature = (
        inference_config_signature(
            cfg
        )
    )

    print(
        f"Detector version: "
        f"{DETECTOR_BUILD_VERSION}",
        flush=True,
    )

    print(
        f"Config signature: "
        f"{config_signature[:16]}...",
        flush=True,
    )

    if work_root.exists() and not args.resume:
        print(
            f"\nRemoving previous work directory: "
            f"{work_root}",
            flush=True,
        )
        shutil.rmtree(
            work_root
        )

    parts_root = (
        work_root
        / "parts"
    )

    parts_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    statuses = []

    print(
        "\nStreaming logical-client processing:",
        flush=True,
    )

    for index, spec in enumerate(
        specs,
        start=1,
    ):
        run_id = str(
            spec["run_id"]
        )
        client_id = str(
            spec["client_id"]
        )

        part_dir = (
            parts_root
            / (
                safe_name(run_id)
                + "__"
                + safe_name(client_id)
            )
        )

        if (
            args.resume
            and part_complete(
                part_dir,
                config_signature,
            )
        ):
            status = json.loads(
                (
                    part_dir
                    / "streaming_part_status.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            print(
                f"[{index:02d}/{len(specs)}] "
                f"{run_id} {client_id} "
                f"RESUME-SKIP "
                f"rounds="
                f"{status.get('round_sample_count', 0)}",
                flush=True,
            )

            statuses.append(
                status
            )
            continue

        print(
            f"[{index:02d}/{len(specs)}] "
            f"{run_id} {client_id} "
            f"manifest_packets="
            f"{int(spec.get('packet_count_manifest', 0)):,}",
            flush=True,
        )

        status = build_one_trace(
            spec=spec,
            ground_truth_rounds=(
                ground_truth_rounds
            ),
            cfg=cfg,
            part_dir=part_dir,
            config_signature=(
                config_signature
            ),
        )

        statuses.append(
            status
        )

        print(
            f"  checkpoint complete: "
            f"round_samples="
            f"{status.get('round_sample_count', 0)}",
            flush=True,
        )

        gc.collect()

    print(
        "\nAll source traces processed. "
        "Assembling final dataset...",
        flush=True,
    )

    result = merge_parts(
        specs=specs,
        work_root=work_root,
        output_dir=output_dir,
        cfg=cfg,
    )

    print(
        "\nROUND DATASET BUILD COMPLETE",
        flush=True,
    )

    print(
        json.dumps(
            result,
            indent=2,
        ),
        flush=True,
    )

    print(
        "\nPer-trace checkpoints retained at:",
        work_root,
        flush=True,
    )


if __name__ == "__main__":
    main()
