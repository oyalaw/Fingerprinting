from __future__ import annotations

import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MODEL_SCHEMA_VERSION = "1.1"
FISHER_TOP_K = 10
LABEL_LEVELS = ("family", "architecture", "variant")
REALTIME_MIN_PACKETS = 20

# Metadata may accompany features for grouping/alignment but never enters X.
MODEL_METADATA_FIELDS = {
    "row_id",
    "experiment_id",
    "client_capture_id",
    "resolved_client_id",
    "row_type",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "window_size_sec",
    "trace_start_offset_sec",
    "trace_end_offset_sec",
    "window_start_global_sec",
    "window_end_global_sec",
}

# Size-normalized mode deliberately suppresses the dominant absolute model
# footprint while preserving rates, fractions, distributions, and timing.
SIZE_NORMALIZED_EXACT_DROP = {
    "packet_count_total",
    "packet_count_up",
    "packet_count_down",
    "packet_count_unknown",
    "bytes_total",
    "bytes_up",
    "bytes_down",
    "bytes_unknown",
    "direction_switch_count",
    "burst_count_total",
    "burst_count_up",
    "burst_count_down",
    "idle_gap_count",
    "tcp_packet_count",
    "udp_packet_count",
    "tcp_syn_count",
    "tcp_ack_count",
    "tcp_fin_count",
    "tcp_rst_count",
    "tcp_retransmission_count",
    "tls_record_count",
    "connection_count",
}


class ArchitectureModelError(RuntimeError):
    pass


def _require_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
        )
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as exc:
        raise ArchitectureModelError(
            "Architecture model training requires scikit-learn. "
            "Install the project dependencies with `python -m pip install -e .`."
        ) from exc
    return {
        "RandomForestClassifier": RandomForestClassifier,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "log_loss": log_loss,
        "StratifiedGroupKFold": StratifiedGroupKFold,
    }


def _numeric(value: Any) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def candidate_feature_columns(
    fieldnames: Sequence[str],
    feature_mode: str,
) -> List[str]:
    if feature_mode not in {"full", "size_normalized"}:
        raise ArchitectureModelError(
            "feature_mode must be full or size_normalized"
        )

    columns = [
        name for name in fieldnames
        if name not in MODEL_METADATA_FIELDS
    ]

    if feature_mode == "size_normalized":
        columns = [
            name for name in columns
            if name not in SIZE_NORMALIZED_EXACT_DROP
        ]

    if not columns:
        raise ArchitectureModelError(
            "No predictor columns remain after feature selection"
        )
    return columns


def read_xy(
    x_csv: str | Path,
    y_csv: str | Path,
) -> tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    with Path(x_csv).open(newline="", encoding="utf-8") as handle:
        x_rows = list(csv.DictReader(handle))

    with Path(y_csv).open(newline="", encoding="utf-8") as handle:
        y_rows = list(csv.DictReader(handle))

    y_by_id = {
        str(row["row_id"]): row
        for row in y_rows
        if row.get("row_id")
    }
    return x_rows, y_by_id


def _joined_rows(
    x_rows: Sequence[Mapping[str, Any]],
    y_by_id: Mapping[str, Mapping[str, Any]],
    row_type: str,
    window_size_sec: Optional[float],
) -> List[Dict[str, Any]]:
    joined: List[Dict[str, Any]] = []
    for x in x_rows:
        if str(x.get("row_type", "")) != row_type:
            continue

        if row_type == "window":
            if _numeric(x.get("packet_count_total")) < REALTIME_MIN_PACKETS:
                continue
            current = _numeric(x.get("window_size_sec"))
            if current <= 0:
                current = max(
                    0.0,
                    _numeric(x.get("window_end_sec"))
                    - _numeric(x.get("window_start_sec")),
                )
            if window_size_sec is None or not math.isclose(
                current,
                float(window_size_sec),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                continue

        row_id = str(x.get("row_id", ""))
        y = y_by_id.get(row_id)
        if y is None:
            continue

        row = dict(x)
        row.update(
            {
                "_family": str(y.get("family", "")).strip(),
                "_architecture": str(
                    y.get("architecture", "")
                ).strip(),
                "_variant": str(y.get("variant", "")).strip(),
            }
        )
        if not all(row[f"_{level}"] for level in LABEL_LEVELS):
            continue
        joined.append(row)

    return joined


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [
            [_numeric(row.get(column)) for column in feature_columns]
            for row in rows
        ],
        dtype=np.float64,
    )


def fisher_score_ranking(
    X: np.ndarray,
    labels: Sequence[str],
    feature_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    """Rank features using the multiclass Fisher score.

    The score is between-class dispersion divided by within-class
    dispersion. It is computed only from predictor features and labels;
    experiment IDs and endpoint identity never enter the calculation.
    """
    if X.ndim != 2 or X.shape[1] != len(feature_columns):
        raise ArchitectureModelError("Invalid feature matrix for Fisher score")
    y = np.asarray(labels, dtype=object)
    classes = sorted(set(str(value) for value in labels))
    if len(classes) < 2:
        return [
            {"feature": name, "score": 0.0}
            for name in feature_columns
        ]

    overall = np.mean(X, axis=0)
    numerator = np.zeros(X.shape[1], dtype=np.float64)
    denominator = np.zeros(X.shape[1], dtype=np.float64)
    for label in classes:
        mask = y == label
        subset = X[mask]
        if subset.size == 0:
            continue
        mean = np.mean(subset, axis=0)
        variance = np.var(subset, axis=0)
        numerator += subset.shape[0] * np.square(mean - overall)
        denominator += subset.shape[0] * variance

    scores = numerator / np.maximum(denominator, 1e-12)
    ranking = sorted(
        (
            {"feature": name, "score": float(score)}
            for name, score in zip(feature_columns, scores)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return ranking


def _select_fisher_features(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    target_key: str,
    top_k: int = FISHER_TOP_K,
) -> tuple[List[str], List[Dict[str, Any]]]:
    labels = [str(row[target_key]) for row in rows]
    X = _matrix(rows, feature_columns)
    ranking = fisher_score_ranking(X, labels, feature_columns)
    keep = max(1, min(int(top_k), len(ranking)))
    selected = [item["feature"] for item in ranking[:keep]]
    return selected, ranking


def _groups_per_class(
    labels: Sequence[str],
    groups: Sequence[str],
) -> Dict[str, int]:
    mapping: Dict[str, set[str]] = defaultdict(set)
    for label, group in zip(labels, groups):
        mapping[str(label)].add(str(group))
    return {
        label: len(values)
        for label, values in mapping.items()
    }


def _evaluate_grouped(
    X: np.ndarray,
    y: Sequence[str],
    groups: Sequence[str],
    classifier_factory,
) -> Dict[str, Any]:
    sk = _require_sklearn()
    classes = sorted(set(y))
    group_counts = _groups_per_class(y, groups)
    min_groups = min(group_counts.values()) if group_counts else 0

    if len(classes) < 2 or min_groups < 2:
        return {
            "status": "insufficient_independent_runs",
            "classes": classes,
            "independent_groups_per_class": group_counts,
        }

    n_splits = min(5, min_groups)
    splitter = sk["StratifiedGroupKFold"](
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    metric_values: Dict[str, List[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
        "log_loss": [],
    }
    y_arr = np.asarray(y, dtype=object)
    groups_arr = np.asarray(groups, dtype=object)
    class_index = {label: index for index, label in enumerate(classes)}

    for train_idx, test_idx in splitter.split(X, y_arr, groups_arr):
        model = classifier_factory()
        model.fit(X[train_idx], y_arr[train_idx])
        truth = y_arr[test_idx]
        prediction = model.predict(X[test_idx])
        raw_probability = model.predict_proba(X[test_idx])
        aligned = np.full(
            (len(test_idx), len(classes)),
            1e-15,
            dtype=np.float64,
        )
        for source_index, label in enumerate(model.classes_):
            aligned[:, class_index[str(label)]] = raw_probability[:, source_index]
        aligned /= aligned.sum(axis=1, keepdims=True)

        metric_values["accuracy"].append(
            float(sk["accuracy_score"](truth, prediction))
        )
        metric_values["balanced_accuracy"].append(
            float(sk["balanced_accuracy_score"](truth, prediction))
        )
        metric_values["macro_precision"].append(
            float(sk["precision_score"](
                truth, prediction, average="macro", zero_division=0
            ))
        )
        metric_values["macro_recall"].append(
            float(sk["recall_score"](
                truth, prediction, average="macro", zero_division=0
            ))
        )
        metric_values["macro_f1"].append(
            float(sk["f1_score"](
                truth, prediction, average="macro", zero_division=0
            ))
        )
        metric_values["log_loss"].append(
            float(sk["log_loss"](truth, aligned, labels=classes))
        )

    result: Dict[str, Any] = {
        "status": "evaluated",
        "folds": n_splits,
        "independent_groups_per_class": group_counts,
    }
    for name, values in metric_values.items():
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))
    return result


def _train_stage(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    target_key: str,
    fisher_top_k: int = FISHER_TOP_K,
) -> Dict[str, Any]:
    sk = _require_sklearn()
    RandomForestClassifier = sk["RandomForestClassifier"]

    labels = [str(row[target_key]) for row in rows]
    classes = sorted(set(labels))
    if not classes:
        return {"kind": "unavailable", "classes": []}
    if len(classes) == 1:
        return {
            "kind": "constant",
            "value": classes[0],
            "classes": classes,
            "sample_count": len(rows),
            "feature_columns": [],
            "fisher_ranking": [],
        }

    selected_columns, fisher_ranking = _select_fisher_features(
        rows,
        feature_columns,
        target_key,
        top_k=fisher_top_k,
    )
    X = _matrix(rows, selected_columns)
    groups = [str(row.get("experiment_id", "")) for row in rows]

    def factory():
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    evaluation = _evaluate_grouped(X, labels, groups, factory)

    model = factory()
    model.fit(X, labels)

    importances = getattr(model, "feature_importances_", None)
    top_features: List[Dict[str, Any]] = []
    if importances is not None:
        ranked = sorted(
            zip(selected_columns, importances),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        top_features = [
            {"feature": name, "importance": float(score)}
            for name, score in ranked[:25]
        ]

    return {
        "kind": "classifier",
        "classes": [str(value) for value in model.classes_],
        "model": model,
        "sample_count": len(rows),
        "independent_experiment_count": len(set(groups)),
        "evaluation": evaluation,
        "feature_selection": "fisher_score",
        "fisher_top_k": len(selected_columns),
        "feature_columns": list(selected_columns),
        "fisher_ranking": fisher_ranking,
        "top_features": top_features,
    }


def train_hierarchy_bundle(
    x_csv: str | Path,
    y_csv: str | Path,
    output_dir: str | Path,
    *,
    mode: str,
    feature_mode: str,
    window_size_sec: Optional[float] = None,
    fisher_top_k: int = FISHER_TOP_K,
) -> Dict[str, Any]:
    if mode not in {"final", "realtime"}:
        raise ArchitectureModelError("mode must be final or realtime")
    if mode == "realtime" and window_size_sec is None:
        raise ArchitectureModelError(
            "realtime model requires window_size_sec"
        )

    x_rows, y_by_id = read_xy(x_csv, y_csv)
    joined = _joined_rows(
        x_rows,
        y_by_id,
        row_type="overall" if mode == "final" else "window",
        window_size_sec=window_size_sec,
    )
    if not joined:
        raise ArchitectureModelError(
            f"No joined samples were found for mode={mode}, "
            f"window_size_sec={window_size_sec}"
        )

    feature_columns = candidate_feature_columns(
        [
            key
            for key in joined[0].keys()
            if not key.startswith("_")
        ],
        feature_mode,
    )

    stages: Dict[str, Any] = {}
    stages["family"] = _train_stage(
        joined,
        feature_columns,
        "_family",
        fisher_top_k=fisher_top_k,
    )

    architecture_by_family: Dict[str, Any] = {}
    families = sorted({row["_family"] for row in joined})
    for family in families:
        subset = [
            row for row in joined
            if row["_family"] == family
        ]
        architecture_by_family[family] = _train_stage(
            subset,
            feature_columns,
            "_architecture",
            fisher_top_k=fisher_top_k,
        )
    stages["architecture_by_family"] = architecture_by_family

    variant_by_parent: Dict[str, Any] = {}
    parents = sorted(
        {
            (row["_family"], row["_architecture"])
            for row in joined
        }
    )
    for family, architecture in parents:
        subset = [
            row for row in joined
            if row["_family"] == family
            and row["_architecture"] == architecture
        ]
        key = f"{family}::{architecture}"
        variant_by_parent[key] = _train_stage(
            subset,
            feature_columns,
            "_variant",
            fisher_top_k=fisher_top_k,
        )
    stages["variant_by_parent"] = variant_by_parent

    bundle = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "mode": mode,
        "feature_mode": feature_mode,
        "window_size_sec": (
            float(window_size_sec)
            if window_size_sec is not None
            else None
        ),
        "feature_columns": list(feature_columns),
        "feature_selection": "fisher_score",
        "fisher_top_k": int(fisher_top_k),
        "sample_count": len(joined),
        "realtime_min_packets": (
            REALTIME_MIN_PACKETS if mode == "realtime" else None
        ),
        "experiment_count": len(
            {
                str(row.get("experiment_id", ""))
                for row in joined
            }
        ),
        "stages": stages,
    }

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    bundle_path = target / "bundle.pkl"
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    metadata = bundle_metadata(bundle)
    metadata_path = target / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "bundle_path": str(bundle_path),
        "metadata_path": str(metadata_path),
        **metadata,
    }


def _stage_metadata(stage: Mapping[str, Any]) -> Dict[str, Any]:
    result = {
        key: value
        for key, value in stage.items()
        if key != "model"
    }
    return result


def bundle_metadata(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    stages = bundle["stages"]
    return {
        "schema_version": bundle["schema_version"],
        "mode": bundle["mode"],
        "feature_mode": bundle["feature_mode"],
        "window_size_sec": bundle["window_size_sec"],
        "feature_columns": bundle["feature_columns"],
        "feature_selection": bundle.get("feature_selection", "none"),
        "fisher_top_k": bundle.get("fisher_top_k"),
        "sample_count": bundle["sample_count"],
        "realtime_min_packets": bundle.get(
            "realtime_min_packets"
        ),
        "experiment_count": bundle["experiment_count"],
        "stages": {
            "family": _stage_metadata(stages["family"]),
            "architecture_by_family": {
                key: _stage_metadata(stage)
                for key, stage in stages[
                    "architecture_by_family"
                ].items()
            },
            "variant_by_parent": {
                key: _stage_metadata(stage)
                for key, stage in stages[
                    "variant_by_parent"
                ].items()
            },
        },
    }


def load_bundle(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ArchitectureModelError(
            "Unsupported architecture model bundle schema"
        )
    return bundle


def _predict_stage(
    stage: Mapping[str, Any],
    feature_row: Mapping[str, Any],
) -> Dict[str, Any]:
    kind = stage.get("kind")
    if kind == "constant":
        value = str(stage["value"])
        return {
            "label": value,
            "confidence": 1.0,
            "probabilities": {value: 1.0},
            "kind": "constant",
        }
    if kind != "classifier":
        return {
            "label": None,
            "confidence": 0.0,
            "probabilities": {},
            "kind": str(kind or "unavailable"),
        }

    feature_columns = stage.get("feature_columns") or []
    if not feature_columns:
        raise ArchitectureModelError(
            "Classifier stage is missing Fisher-selected feature columns"
        )
    vector = np.asarray(
        [_numeric(feature_row.get(name)) for name in feature_columns],
        dtype=np.float64,
    )
    model = stage["model"]
    probabilities = model.predict_proba(vector.reshape(1, -1))[0]
    classes = [str(value) for value in model.classes_]
    distribution = {
        label: float(probability)
        for label, probability in zip(classes, probabilities)
    }
    label = max(distribution, key=distribution.get)
    return {
        "label": label,
        "confidence": float(distribution[label]),
        "probabilities": distribution,
        "kind": "classifier",
    }


def predict_hierarchy(
    bundle: Mapping[str, Any],
    feature_row: Mapping[str, Any],
) -> Dict[str, Any]:
    stages = bundle["stages"]

    family = _predict_stage(stages["family"], feature_row)
    family_label = family.get("label")

    architecture = {
        "label": None,
        "confidence": 0.0,
        "probabilities": {},
        "kind": "unavailable",
    }
    if family_label:
        stage = stages["architecture_by_family"].get(family_label)
        if stage is not None:
            architecture = _predict_stage(stage, feature_row)

    variant = {
        "label": None,
        "confidence": 0.0,
        "probabilities": {},
        "kind": "unavailable",
    }
    architecture_label = architecture.get("label")
    if family_label and architecture_label:
        stage = stages["variant_by_parent"].get(
            f"{family_label}::{architecture_label}"
        )
        if stage is not None:
            variant = _predict_stage(stage, feature_row)

    return {
        "family": family,
        "architecture": architecture,
        "variant": variant,
    }


def model_directory_name(
    mode: str,
    feature_mode: str,
    window_size_sec: Optional[float] = None,
) -> str:
    if mode == "final":
        suffix = "final"
    else:
        value = float(window_size_sec or 0.0)
        token = ("%g" % value).replace(".", "p")
        suffix = f"realtime_{token}s"
    return f"{feature_mode}/{suffix}"


def discover_bundle(
    model_root: str | Path,
    *,
    mode: str,
    feature_mode: str,
    window_size_sec: Optional[float] = None,
) -> Optional[Path]:
    path = (
        Path(model_root)
        / model_directory_name(
            mode,
            feature_mode,
            window_size_sec,
        )
        / "bundle.pkl"
    )
    return path if path.exists() else None
