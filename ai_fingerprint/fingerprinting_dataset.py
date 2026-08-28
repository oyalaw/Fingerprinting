from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


class FingerprintingDataError(RuntimeError):
    pass


# These are the only labels copied out of client/server ground-truth logs.
# They are targets (Y), never predictor features (X).
GROUND_TRUTH_LABEL_FIELDS = [
    "task",
    "deployment",
    "framework",
    "runtime",
    "family",
    "architecture",
    "variant",
    "application",
    "device",
]

OPTIONAL_CONTEXT_LABEL_FIELDS = [
    "dataset",
    "dataset_split",
    "operating_system",
    "precision",
    "batch_size",
    "input_size",
]

# Metadata may accompany X for grouping/splitting, but it is not part of the
# model's predictor matrix.
PROXY_FEATURE_METADATA_FIELDS = [
    "experiment_id",
    "client_capture_id",
    "row_type",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "window_size_sec",
    "trace_start_offset_sec",
    "trace_end_offset_sec",
    "window_start_global_sec",
    "window_end_global_sec",
]

# Raw packet identifiers are observable on the wire, but they are intentionally
# excluded from the classifier-ready sequence because IP/port identity can
# create shortcut learning and undermine cross-network generalization.
SEQUENCE_IDENTITY_FIELDS = {
    "timestamp_epoch",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
}

SAFE_SEQUENCE_FIELDS = [
    "experiment_id",
    "packet_index",
    "relative_time_sec",
    "frame_length",
    "direction",
    "transport_protocol",
    "tcp_flags_hex",
    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "retransmission",
    "tls_record_count",
    "tls_record_lengths",
]

# Exact fields that must never appear in attacker predictor input.
FORBIDDEN_PREDICTOR_FIELDS = {
    "role",
    "event",
    "timestamp_utc",
    "framework",
    "runtime",
    "family",
    "architecture",
    "variant",
    "application",
    "dataset",
    "dataset_split",
    "device",
    "operating_system",
    "task",
    "deployment",
    "execution_mode",
    "precision",
    "batch_size",
    "input_size",
    "client_id",
    "round",
    "epoch",
    "local_epoch",
    "global_step",
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mse",
    "mae",
    "kl_loss",
    "reconstruction_loss",
    "learning_rate",
    "samples_processed",
    "update_norm_l2",
    "global_update_norm_l2",
    "global_model_norm_l2",
    "local_model_norm_l2",
    "client_model_norm_l2",
    "client_update_norm_l2",
    "round_duration_sec",
    "aggregation_time_ms",
    "evaluation_time_ms",
}

# Resource/system telemetry is client/server characterization only.
FORBIDDEN_PREDICTOR_PREFIXES = (
    "cpu_",
    "gpu_",
    "memory_",
    "system_memory_",
    "power_",
    "energy_",
    "process_cpu_",
    "system_cpu_",
)

# Additional resource names that do not follow the prefix convention.
FORBIDDEN_RESOURCE_FIELDS = {
    "cpu_usage_percent",
    "gpu_usage_percent",
    "memory_usage_mb",
    "memory_usage_percent",
    "system_memory_usage_percent",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
    "cpu_power_w",
    "gpu_power_w",
    "system_power_w",
    "cpu_energy_j",
    "gpu_energy_j",
    "system_energy_j",
    "telemetry_source",
    "sample_interval_ms",
    "network_interface",
}


def is_forbidden_predictor_field(name: str) -> bool:
    normalized = str(name).strip().lower()
    if normalized in FORBIDDEN_PREDICTOR_FIELDS:
        return True
    if normalized in FORBIDDEN_RESOURCE_FIELDS:
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in FORBIDDEN_PREDICTOR_PREFIXES
    )


def validate_proxy_feature_columns(
    fieldnames: Sequence[str] | None,
) -> List[str]:
    if not fieldnames:
        raise FingerprintingDataError(
            "Proxy feature CSV has no header"
        )

    columns = [str(name) for name in fieldnames]
    if "experiment_id" not in columns:
        raise FingerprintingDataError(
            "Proxy feature CSV must contain experiment_id"
        )

    forbidden = sorted(
        name
        for name in columns
        if is_forbidden_predictor_field(name)
    )
    if forbidden:
        raise FingerprintingDataError(
            "Attacker predictor input contains forbidden client/server "
            f"or resource fields: {forbidden}"
        )

    predictors = [
        name
        for name in columns
        if name not in PROXY_FEATURE_METADATA_FIELDS
    ]
    if not predictors:
        raise FingerprintingDataError(
            "No network predictor columns remain after metadata removal"
        )

    return predictors


def _parse_numeric(value: Any, field: str, row_number: int) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FingerprintingDataError(
            f"Proxy predictor {field!r} is not numeric at row "
            f"{row_number}: {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise FingerprintingDataError(
            f"Proxy predictor {field!r} is non-finite at row "
            f"{row_number}: {value!r}"
        )
    return result


def sanitize_packet_sequence(
    raw_packet_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    source = Path(raw_packet_csv)
    target = Path(output_csv)
    if not source.exists():
        raise FingerprintingDataError(
            f"Raw packet sequence not found: {source}"
        )

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        missing = [
            name
            for name in SAFE_SEQUENCE_FIELDS
            if name not in fieldnames
        ]
        if missing:
            raise FingerprintingDataError(
                "Raw packet sequence is missing required fields: "
                f"{missing}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as out:
            writer = csv.DictWriter(
                out,
                fieldnames=SAFE_SEQUENCE_FIELDS,
            )
            writer.writeheader()
            for row in reader:
                writer.writerow(
                    {
                        field: row.get(field, "")
                        for field in SAFE_SEQUENCE_FIELDS
                    }
                )

    return target


def _read_ground_truth_indices(
    paths: Sequence[str | Path],
) -> tuple[
    Dict[str, Dict[str, Any]],
    Dict[tuple[str, str], Dict[str, Any]],
]:
    """Index ground truth by experiment and, when available, client ID.

    Shared workload labels are experiment-level. Client device/context labels
    are indexed by ``client_id`` so a multi-client proxy capture can be split
    into per-client feature files without collapsing different client devices
    into one ambiguous experiment label.
    """
    shared_values: Dict[str, Dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    client_values_any: Dict[str, Dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    client_values_by_id: Dict[
        str,
        Dict[str, Dict[str, set[str]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    if not paths:
        raise FingerprintingDataError(
            "At least one client/server ground-truth JSONL file is required"
        )

    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FingerprintingDataError(
                f"Ground-truth file not found: {path}"
            )
        if "_resource" in path.name.lower():
            raise FingerprintingDataError(
                "Resource telemetry files cannot be used as "
                f"fingerprinting labels or predictors: {path}"
            )

        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FingerprintingDataError(
                        f"Invalid JSON in {path} line {line_number}"
                    ) from exc

                experiment_id = str(
                    record.get("experiment_id", "")
                ).strip()
                if not experiment_id:
                    raise FingerprintingDataError(
                        f"Missing experiment_id in {path} "
                        f"line {line_number}"
                    )

                role = str(record.get("role", "")).strip().lower()
                for field in (
                    "task",
                    "deployment",
                    "framework",
                    "runtime",
                    "family",
                    "architecture",
                    "variant",
                    "application",
                ):
                    value = record.get(field)
                    if value is not None and str(value) != "":
                        shared_values[experiment_id][field].add(
                            json.dumps(value, sort_keys=True)
                        )

                if role == "client":
                    client_id = str(
                        record.get("client_id", "")
                    ).strip()
                    for field in (
                        "device",
                        *OPTIONAL_CONTEXT_LABEL_FIELDS,
                    ):
                        value = record.get(field)
                        if value is None or str(value) == "":
                            continue
                        encoded = json.dumps(value, sort_keys=True)
                        client_values_any[experiment_id][field].add(
                            encoded
                        )
                        if client_id:
                            client_values_by_id[experiment_id][client_id][
                                field
                            ].add(encoded)

    shared_resolved: Dict[str, Dict[str, Any]] = {}
    for experiment_id in set(shared_values) | set(client_values_any):
        row: Dict[str, Any] = {"experiment_id": experiment_id}
        for field in (
            "task",
            "deployment",
            "framework",
            "runtime",
            "family",
            "architecture",
            "variant",
            "application",
        ):
            choices = shared_values[experiment_id].get(field, set())
            if not choices:
                raise FingerprintingDataError(
                    f"Ground truth for {experiment_id!r} is missing "
                    f"required label {field!r}"
                )
            if len(choices) != 1:
                raise FingerprintingDataError(
                    f"Conflicting client/server ground-truth values for "
                    f"{experiment_id!r}/{field!r}: {sorted(choices)}"
                )
            row[field] = json.loads(next(iter(choices)))
        shared_resolved[experiment_id] = row

    experiment_labels: Dict[str, Dict[str, Any]] = {}
    for experiment_id, shared in shared_resolved.items():
        device_choices = client_values_any[experiment_id].get(
            "device", set()
        )
        if len(device_choices) == 1:
            row = dict(shared)
            row["device"] = json.loads(next(iter(device_choices)))
            for field in OPTIONAL_CONTEXT_LABEL_FIELDS:
                choices = client_values_any[experiment_id].get(
                    field, set()
                )
                if len(choices) == 1:
                    row[field] = json.loads(next(iter(choices)))
            experiment_labels[experiment_id] = row

    client_labels: Dict[tuple[str, str], Dict[str, Any]] = {}
    for experiment_id, clients in client_values_by_id.items():
        shared = shared_resolved.get(experiment_id)
        if shared is None:
            continue
        for client_id, values in clients.items():
            device_choices = values.get("device", set())
            if not device_choices:
                continue
            if len(device_choices) != 1:
                raise FingerprintingDataError(
                    f"Conflicting device values for "
                    f"{experiment_id!r}/{client_id!r}: "
                    f"{sorted(device_choices)}"
                )
            row = dict(shared)
            row["client_id"] = client_id
            row["device"] = json.loads(next(iter(device_choices)))
            for field in OPTIONAL_CONTEXT_LABEL_FIELDS:
                choices = values.get(field, set())
                if len(choices) == 1:
                    row[field] = json.loads(next(iter(choices)))
                elif len(choices) > 1:
                    raise FingerprintingDataError(
                        f"Conflicting client contextual values for "
                        f"{experiment_id!r}/{client_id!r}/{field!r}: "
                        f"{sorted(choices)}"
                    )
            client_labels[(experiment_id, client_id)] = row

    return experiment_labels, client_labels


def _read_ground_truth_labels(
    paths: Sequence[str | Path],
) -> Dict[str, Dict[str, Any]]:
    """Backward-compatible single-client experiment label resolver."""
    experiment_labels, client_labels = _read_ground_truth_indices(paths)
    all_experiments = {
        experiment_id
        for experiment_id, _ in client_labels
    }
    for experiment_id in all_experiments:
        if experiment_id not in experiment_labels:
            clients = sorted(
                client_id
                for exp, client_id in client_labels
                if exp == experiment_id
            )
            raise FingerprintingDataError(
                f"Multiple client labels exist for {experiment_id!r}: "
                f"{clients}. Use per-client proxy feature files containing "
                "client_capture_id that matches the federated client_id."
            )
    return experiment_labels

def build_fingerprinting_dataset(
    proxy_feature_csvs: Sequence[str | Path],
    ground_truth_jsonls: Sequence[str | Path],
    output_dir: str | Path,
    prefix: str = "fingerprinting",
    client_capture_id_map: Mapping[
        tuple[str, str], str
    ] | None = None,
) -> Dict[str, Any]:
    """
    Build physically separated X and Y files.

    X contains only proxy-observable handcrafted network predictors plus
    grouping metadata. Y contains only labels recovered from client/server
    ground truth. Resource telemetry is never read by this function.
    """
    if not proxy_feature_csvs:
        raise FingerprintingDataError(
            "At least one proxy feature CSV is required"
        )

    experiment_labels, client_labels = _read_ground_truth_indices(
        ground_truth_jsonls
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    x_path = output / f"{prefix}_X_proxy.csv"
    y_path = output / f"{prefix}_Y_ground_truth.csv"
    schema_path = output / f"{prefix}_schema.json"

    x_rows: List[Dict[str, Any]] = []
    y_rows: List[Dict[str, Any]] = []
    predictor_columns: List[str] | None = None
    metadata_columns: List[str] | None = None
    row_id = 0

    for csv_value in proxy_feature_csvs:
        path = Path(csv_value)
        if not path.exists():
            raise FingerprintingDataError(
                f"Proxy feature CSV not found: {path}"
            )

        if "_resource" in path.name.lower():
            raise FingerprintingDataError(
                f"Resource telemetry is not a proxy feature source: {path}"
            )

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            current_predictors = validate_proxy_feature_columns(
                reader.fieldnames
            )
            current_metadata = [
                field
                for field in PROXY_FEATURE_METADATA_FIELDS
                if field in (reader.fieldnames or [])
            ]

            if predictor_columns is None:
                predictor_columns = current_predictors
                metadata_columns = current_metadata
            elif current_predictors != predictor_columns:
                raise FingerprintingDataError(
                    "Proxy feature schemas differ across files. "
                    f"Expected {predictor_columns}, got "
                    f"{current_predictors} in {path}"
                )

            for source_row_number, row in enumerate(
                reader,
                start=2,
            ):
                experiment_id = str(
                    row.get("experiment_id", "")
                ).strip()
                client_capture_id = str(
                    row.get("client_capture_id", "") or ""
                ).strip()

                resolved_client_id = ""
                if client_capture_id:
                    capture_key = (
                        experiment_id,
                        client_capture_id,
                    )
                    resolved_client_id = (
                        (client_capture_id_map or {}).get(
                            capture_key,
                            client_capture_id,
                        )
                    )
                    label_key = (
                        experiment_id,
                        resolved_client_id,
                    )
                    if label_key not in client_labels:
                        available = sorted(
                            client_id
                            for exp, client_id in client_labels
                            if exp == experiment_id
                        )
                        raise FingerprintingDataError(
                            f"No client ground truth found for proxy "
                            f"sample {experiment_id!r}/"
                            f"{client_capture_id!r}. Resolved client ID: "
                            f"{resolved_client_id!r}. Available federated "
                            f"client IDs: {available}. For new runs, keep "
                            "the proxy manifest and client "
                            "network_registration events together so the "
                            "mapping can be resolved automatically."
                        )
                    label_row_source = client_labels[label_key]
                else:
                    if experiment_id not in experiment_labels:
                        available = sorted(
                            client_id
                            for exp, client_id in client_labels
                            if exp == experiment_id
                        )
                        if available:
                            raise FingerprintingDataError(
                                f"Experiment {experiment_id!r} contains "
                                f"multiple clients {available}. Use the "
                                "per-client proxy feature files rather than "
                                "the combined multi-client feature file."
                            )
                        raise FingerprintingDataError(
                            f"No ground truth found for proxy experiment "
                            f"{experiment_id!r} from {path}"
                        )
                    label_row_source = experiment_labels[experiment_id]

                if label_row_source["deployment"] == "local":
                    raise FingerprintingDataError(
                        f"Experiment {experiment_id!r} is labeled local. "
                        "Local inference/training has no workload exchange "
                        "for a network-side proxy to fingerprint. Do not "
                        "train the network classifier on unrelated local "
                        "traffic."
                    )

                row_id += 1
                x_row: Dict[str, Any] = {
                    "row_id": row_id,
                }
                for field in current_metadata:
                    x_row[field] = row.get(field, "")

                for field in current_predictors:
                    x_row[field] = _parse_numeric(
                        row.get(field),
                        field=field,
                        row_number=source_row_number,
                    )

                y_row = {
                    "row_id": row_id,
                    "experiment_id": experiment_id,
                }
                if client_capture_id:
                    y_row["client_capture_id"] = client_capture_id
                    y_row["resolved_client_id"] = (
                        resolved_client_id
                    )
                for field in GROUND_TRUTH_LABEL_FIELDS:
                    y_row[field] = label_row_source[field]
                for field in OPTIONAL_CONTEXT_LABEL_FIELDS:
                    if field in label_row_source:
                        y_row[field] = label_row_source[field]

                x_rows.append(x_row)
                y_rows.append(y_row)

    if not x_rows:
        raise FingerprintingDataError(
            "No proxy feature rows were found"
        )

    assert predictor_columns is not None
    metadata_columns = metadata_columns or ["experiment_id"]

    x_fields = (
        ["row_id"]
        + metadata_columns
        + predictor_columns
    )
    y_fields = [
        "row_id",
        "experiment_id",
        *(["client_capture_id"] if any(
            "client_capture_id" in row for row in y_rows
        ) else []),
        *(["resolved_client_id"] if any(
            "resolved_client_id" in row for row in y_rows
        ) else []),
        *GROUND_TRUTH_LABEL_FIELDS,
        *[
            field
            for field in OPTIONAL_CONTEXT_LABEL_FIELDS
            if any(field in row for row in y_rows)
        ],
    ]

    with x_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=x_fields)
        writer.writeheader()
        writer.writerows(x_rows)

    with y_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=y_fields)
        writer.writeheader()
        writer.writerows(y_rows)

    schema = {
        "schema_version": "1.1",
        "policy": {
            "predictor_source": "proxy_only",
            "ground_truth_source": "client_server_labels_only",
            "resource_telemetry_in_predictors": False,
            "client_server_events_in_predictors": False,
            "experiment_id_is_predictor": False,
            "client_capture_id_is_predictor": False,
            "resolved_client_id_is_predictor": False,
            "global_timing_alignment_is_predictor": False,
            "row_type_is_predictor": False,
            "window_identifiers_are_predictors": False,
        },
        "x_proxy_csv": str(x_path),
        "y_ground_truth_csv": str(y_path),
        "predictor_columns": predictor_columns,
        "metadata_columns": [
            "row_id",
            *metadata_columns,
        ],
        "label_columns": GROUND_TRUTH_LABEL_FIELDS,
        "context_columns": [
            field
            for field in OPTIONAL_CONTEXT_LABEL_FIELDS
            if field in y_fields
        ],
        "forbidden_predictor_fields": sorted(
            FORBIDDEN_PREDICTOR_FIELDS
            | FORBIDDEN_RESOURCE_FIELDS
        ),
        "forbidden_predictor_prefixes": list(
            FORBIDDEN_PREDICTOR_PREFIXES
        ),
        "row_count": len(x_rows),
        "experiment_count": len(
            {row["experiment_id"] for row in y_rows}
        ),
        "split_policy": (
            "Split by experiment/run/session/device grouping as appropriate; "
            "never randomly split packets/windows from the same experiment "
            "across train and test."
        ),
    }
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "X_proxy_csv": str(x_path),
        "Y_ground_truth_csv": str(y_path),
        "schema_json": str(schema_path),
        "row_count": len(x_rows),
        "experiment_count": schema["experiment_count"],
        "predictor_count": len(predictor_columns),
    }


def load_xy(
    x_proxy_csv: str | Path,
    y_ground_truth_csv: str | Path,
    schema_json: str | Path,
    target: str,
) -> tuple[List[List[float]], List[Any], List[str]]:
    """
    Strict loader for downstream classifiers.

    Only schema-declared predictor_columns are returned as X. Metadata columns,
    experiment IDs, labels, and resource telemetry cannot enter X through this
    loader.
    """
    schema = json.loads(
        Path(schema_json).read_text(encoding="utf-8")
    )
    predictors = list(schema["predictor_columns"])

    if target not in schema["label_columns"]:
        raise FingerprintingDataError(
            f"Target {target!r} is not an allowed ground-truth label. "
            f"Choose from {schema['label_columns']}"
        )

    with Path(x_proxy_csv).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        x_reader = csv.DictReader(handle)
        validate_proxy_feature_columns(
            [
                field
                for field in (x_reader.fieldnames or [])
                if field != "row_id"
            ]
        )
        x_by_id = {
            str(row["row_id"]): [
                _parse_numeric(
                    row.get(field),
                    field,
                    row_number=index,
                )
                for field in predictors
            ]
            for index, row in enumerate(x_reader, start=2)
        }

    with Path(y_ground_truth_csv).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        y_reader = csv.DictReader(handle)
        if target not in (y_reader.fieldnames or []):
            raise FingerprintingDataError(
                f"Target {target!r} is absent from Y"
            )
        y_by_id = {
            str(row["row_id"]): row[target]
            for row in y_reader
        }

    if set(x_by_id) != set(y_by_id):
        raise FingerprintingDataError(
            "X and Y row_id sets do not match"
        )

    ordered = sorted(x_by_id, key=lambda value: int(value))
    X = [x_by_id[row_id] for row_id in ordered]
    y = [y_by_id[row_id] for row_id in ordered]
    return X, y, predictors
