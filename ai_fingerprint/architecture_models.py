from __future__ import annotations

import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MODEL_SCHEMA_VERSION = "1.1"
LABEL_LEVELS = ("family", "architecture", "variant")
REALTIME_MIN_PACKETS = 20

# v0.8.7 stage-specific Fisher selection defaults.
DEFAULT_FISHER_TOP_K = 25
DEFAULT_FISHER_MIN_SCORE = 1e-6
DEFAULT_FISHER_MIN_FEATURES = 8
FISHER_EPSILON = 1e-12

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
        from sklearn.metrics import balanced_accuracy_score, f1_score
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as exc:
        raise ArchitectureModelError(
            "Architecture model training requires scikit-learn. "
            "Install the project dependencies with "
            "`python -m pip install -e .`."
        ) from exc
    return (
        RandomForestClassifier,
        balanced_accuracy_score,
        f1_score,
        StratifiedGroupKFold,
    )


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
        name
        for name in fieldnames
        if name not in MODEL_METADATA_FIELDS
    ]

    if feature_mode == "size_normalized":
        columns = [
            name
            for name in columns
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
            if (
                _numeric(x.get("packet_count_total"))
                < REALTIME_MIN_PACKETS
            ):
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
        if not all(
            row[f"_{level}"]
            for level in LABEL_LEVELS
        ):
            continue
        joined.append(row)

    return joined


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [
            [
                _numeric(row.get(column))
                for column in feature_columns
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


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


def fisher_score_ranking(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    target_key: str,
) -> List[Dict[str, Any]]:
    """Compute a class-balanced multiclass Fisher score for every feature.

    The score is:

        sum_c (mu_c - mean(mu_c))^2
        --------------------------------
        sum_c var_c + epsilon

    Each class contributes equally regardless of its number of windows. This
    prevents long runs/classes with many active windows from dominating the
    univariate ranking.

    The score measures separability, not environmental invariance. It should
    therefore be paired with run/device/network-condition generalization
    experiments.
    """
    labels = [str(row[target_key]) for row in rows]
    classes = sorted(set(labels))
    if len(classes) < 2:
        return []

    ranking: List[Dict[str, Any]] = []
    for feature in feature_columns:
        class_means: List[float] = []
        class_variances: List[float] = []
        class_counts: Dict[str, int] = {}

        for label in classes:
            values = np.asarray(
                [
                    _numeric(row.get(feature))
                    for row in rows
                    if str(row[target_key]) == label
                ],
                dtype=np.float64,
            )
            class_counts[label] = int(values.size)
            if values.size == 0:
                continue
            class_means.append(float(np.mean(values)))
            class_variances.append(float(np.var(values)))

        if len(class_means) != len(classes):
            score = 0.0
        else:
            grand_mean = float(np.mean(class_means))
            between = float(
                sum(
                    (value - grand_mean) ** 2
                    for value in class_means
                )
            )
            within = float(sum(class_variances))
            score = between / (within + FISHER_EPSILON)

        ranking.append(
            {
                "feature": feature,
                "fisher_score": float(score),
                "class_counts": class_counts,
            }
        )

    ranking.sort(
        key=lambda item: (
            float(item["fisher_score"]),
            str(item["feature"]),
        ),
        reverse=True,
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
    return ranking


def select_fisher_features(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    target_key: str,
    *,
    top_k: int = DEFAULT_FISHER_TOP_K,
    min_score: float = DEFAULT_FISHER_MIN_SCORE,
    min_features: int = DEFAULT_FISHER_MIN_FEATURES,
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Select a stage-specific feature subset from Fisher scores."""
    ranking = fisher_score_ranking(
        rows,
        feature_columns,
        target_key,
    )
    if not ranking:
        return list(feature_columns), ranking

    top_k = max(1, int(top_k))
    min_features = max(1, int(min_features))
    min_features = min(min_features, len(ranking))

    above_threshold = [
        item
        for item in ranking
        if float(item["fisher_score"]) >= float(min_score)
    ]

    selected_items = above_threshold[:top_k]
    if len(selected_items) < min_features:
        selected_items = ranking[:min_features]

    selected = [
        str(item["feature"])
        for item in selected_items
    ]

    selected_set = set(selected)
    for item in ranking:
        item["selected"] = (
            str(item["feature"]) in selected_set
        )
    return selected, ranking


def _make_classifier():
    RandomForestClassifier, *_ = _require_sklearn()
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _evaluate_grouped_with_fisher(
    rows: Sequence[Mapping[str, Any]],
    candidate_columns: Sequence[str],
    target_key: str,
    *,
    fisher_top_k: int,
    fisher_min_score: float,
    fisher_min_features: int,
) -> Dict[str, Any]:
    (
        _RandomForestClassifier,
        balanced_accuracy_score,
        f1_score,
        StratifiedGroupKFold,
    ) = _require_sklearn()

    labels = [str(row[target_key]) for row in rows]
    groups = [
        str(row.get("experiment_id", ""))
        for row in rows
    ]
    classes = sorted(set(labels))
    group_counts = _groups_per_class(labels, groups)
    min_groups = min(group_counts.values()) if group_counts else 0

    if len(classes) < 2 or min_groups < 2:
        return {
            "status": "insufficient_independent_runs",
            "classes": classes,
            "independent_groups_per_class": group_counts,
            "feature_selection": (
                "Fisher selection is fitted on training folds only "
                "when grouped evaluation is possible."
            ),
        }

    n_splits = min(5, min_groups)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    labels_arr = np.asarray(labels, dtype=object)
    groups_arr = np.asarray(groups, dtype=object)
    row_indices = np.arange(len(rows))

    balanced_scores: List[float] = []
    f1_scores: List[float] = []
    selected_frequency: Dict[str, int] = defaultdict(int)
    fold_selected_counts: List[int] = []

    for train_idx, test_idx in splitter.split(
        row_indices,
        labels_arr,
        groups_arr,
    ):
        train_rows = [rows[int(index)] for index in train_idx]
        test_rows = [rows[int(index)] for index in test_idx]

        selected, _ranking = select_fisher_features(
            train_rows,
            candidate_columns,
            target_key,
            top_k=fisher_top_k,
            min_score=fisher_min_score,
            min_features=fisher_min_features,
        )
        fold_selected_counts.append(len(selected))
        for feature in selected:
            selected_frequency[feature] += 1

        X_train = _matrix(train_rows, selected)
        X_test = _matrix(test_rows, selected)

        model = _make_classifier()
        model.fit(
            X_train,
            labels_arr[train_idx],
        )
        prediction = model.predict(X_test)

        balanced_scores.append(
            float(
                balanced_accuracy_score(
                    labels_arr[test_idx],
                    prediction,
                )
            )
        )
        f1_scores.append(
            float(
                f1_score(
                    labels_arr[test_idx],
                    prediction,
                    average="macro",
                    zero_division=0,
                )
            )
        )

    stability = [
        {
            "feature": feature,
            "selected_folds": int(count),
            "selection_frequency": float(count / n_splits),
        }
        for feature, count in sorted(
            selected_frequency.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
            reverse=True,
        )
    ]

    return {
        "status": "evaluated",
        "folds": n_splits,
        "balanced_accuracy_mean": float(
            np.mean(balanced_scores)
        ),
        "balanced_accuracy_std": float(
            np.std(balanced_scores)
        ),
        "macro_f1_mean": float(
            np.mean(f1_scores)
        ),
        "macro_f1_std": float(
            np.std(f1_scores)
        ),
        "independent_groups_per_class": group_counts,
        "feature_selection": {
            "method": "class_balanced_fisher",
            "fitted_inside_each_training_fold": True,
            "top_k": int(fisher_top_k),
            "min_score": float(fisher_min_score),
            "min_features": int(fisher_min_features),
            "selected_count_mean": float(
                np.mean(fold_selected_counts)
            ),
            "selected_count_min": int(
                min(fold_selected_counts)
            ),
            "selected_count_max": int(
                max(fold_selected_counts)
            ),
            "selection_stability": stability,
        },
    }


def _train_stage(
    rows: Sequence[Mapping[str, Any]],
    candidate_columns: Sequence[str],
    target_key: str,
    *,
    fisher_top_k: int,
    fisher_min_score: float,
    fisher_min_features: int,
) -> Dict[str, Any]:
    labels = [str(row[target_key]) for row in rows]
    classes = sorted(set(labels))
    if not classes:
        return {
            "kind": "unavailable",
            "classes": [],
            "candidate_feature_count": len(candidate_columns),
        }

    if len(classes) == 1:
        return {
            "kind": "constant",
            "value": classes[0],
            "classes": classes,
            "sample_count": len(rows),
            "candidate_feature_count": len(candidate_columns),
            "feature_columns": [],
            "fisher_ranking": [],
            "note": (
                "Only one class is represented at this hierarchy stage; "
                "no discriminative feature selection was performed."
            ),
        }

    selected_features, ranking = select_fisher_features(
        rows,
        candidate_columns,
        target_key,
        top_k=fisher_top_k,
        min_score=fisher_min_score,
        min_features=fisher_min_features,
    )

    groups = [
        str(row.get("experiment_id", ""))
        for row in rows
    ]

    evaluation = _evaluate_grouped_with_fisher(
        rows,
        candidate_columns,
        target_key,
        fisher_top_k=fisher_top_k,
        fisher_min_score=fisher_min_score,
        fisher_min_features=fisher_min_features,
    )

    model = _make_classifier()
    X = _matrix(rows, selected_features)
    model.fit(X, labels)

    importances = getattr(
        model,
        "feature_importances_",
        None,
    )
    top_features: List[Dict[str, Any]] = []
    if importances is not None:
        ranked_importance = sorted(
            zip(selected_features, importances),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        top_features = [
            {
                "feature": name,
                "importance": float(score),
            }
            for name, score in ranked_importance[:25]
        ]

    return {
        "kind": "classifier",
        "classes": [
            str(value)
            for value in model.classes_
        ],
        "model": model,
        "sample_count": len(rows),
        "independent_experiment_count": len(set(groups)),
        "candidate_feature_count": len(candidate_columns),
        "selected_feature_count": len(selected_features),
        "feature_columns": list(selected_features),
        "feature_selection": {
            "method": "class_balanced_fisher",
            "top_k": int(fisher_top_k),
            "min_score": float(fisher_min_score),
            "min_features": int(fisher_min_features),
        },
        "fisher_ranking": ranking,
        "evaluation": evaluation,
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
    fisher_top_k: int = DEFAULT_FISHER_TOP_K,
    fisher_min_score: float = DEFAULT_FISHER_MIN_SCORE,
    fisher_min_features: int = DEFAULT_FISHER_MIN_FEATURES,
) -> Dict[str, Any]:
    if mode not in {"final", "realtime"}:
        raise ArchitectureModelError(
            "mode must be final or realtime"
        )
    if mode == "realtime" and window_size_sec is None:
        raise ArchitectureModelError(
            "realtime model requires window_size_sec"
        )

    x_rows, y_by_id = read_xy(x_csv, y_csv)
    joined = _joined_rows(
        x_rows,
        y_by_id,
        row_type=(
            "overall"
            if mode == "final"
            else "window"
        ),
        window_size_sec=window_size_sec,
    )
    if not joined:
        raise ArchitectureModelError(
            f"No joined samples were found for mode={mode}, "
            f"window_size_sec={window_size_sec}"
        )

    candidate_columns = candidate_feature_columns(
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
        candidate_columns,
        "_family",
        fisher_top_k=fisher_top_k,
        fisher_min_score=fisher_min_score,
        fisher_min_features=fisher_min_features,
    )

    architecture_by_family: Dict[str, Any] = {}
    families = sorted(
        {row["_family"] for row in joined}
    )
    for family in families:
        subset = [
            row
            for row in joined
            if row["_family"] == family
        ]
        architecture_by_family[family] = _train_stage(
            subset,
            candidate_columns,
            "_architecture",
            fisher_top_k=fisher_top_k,
            fisher_min_score=fisher_min_score,
            fisher_min_features=fisher_min_features,
        )
    stages[
        "architecture_by_family"
    ] = architecture_by_family

    variant_by_parent: Dict[str, Any] = {}
    parents = sorted(
        {
            (
                row["_family"],
                row["_architecture"],
            )
            for row in joined
        }
    )
    for family, architecture in parents:
        subset = [
            row
            for row in joined
            if row["_family"] == family
            and row["_architecture"] == architecture
        ]
        key = f"{family}::{architecture}"
        variant_by_parent[key] = _train_stage(
            subset,
            candidate_columns,
            "_variant",
            fisher_top_k=fisher_top_k,
            fisher_min_score=fisher_min_score,
            fisher_min_features=fisher_min_features,
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
        # Retained for provenance/backward inspection. Prediction uses the
        # stage-specific feature_columns stored inside each trained stage.
        "feature_columns": list(candidate_columns),
        "candidate_feature_columns": list(
            candidate_columns
        ),
        "feature_selection": {
            "method": "class_balanced_fisher",
            "stage_specific": True,
            "top_k": int(fisher_top_k),
            "min_score": float(fisher_min_score),
            "min_features": int(fisher_min_features),
            "evaluation_leakage_control": (
                "Fisher selection is re-fitted only on each training fold "
                "during grouped cross-validation."
            ),
        },
        "sample_count": len(joined),
        "realtime_min_packets": (
            REALTIME_MIN_PACKETS
            if mode == "realtime"
            else None
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
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fisher_csv_path = target / "fisher_scores.csv"
    _write_fisher_scores_csv(
        metadata,
        fisher_csv_path,
    )

    return {
        "bundle_path": str(bundle_path),
        "metadata_path": str(metadata_path),
        "fisher_scores_csv": str(fisher_csv_path),
        **metadata,
    }


def _stage_metadata(
    stage: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        key: value
        for key, value in stage.items()
        if key != "model"
    }


def bundle_metadata(
    bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    stages = bundle["stages"]
    return {
        "schema_version": bundle["schema_version"],
        "mode": bundle["mode"],
        "feature_mode": bundle["feature_mode"],
        "window_size_sec": bundle[
            "window_size_sec"
        ],
        "feature_columns": bundle[
            "feature_columns"
        ],
        "candidate_feature_columns": bundle.get(
            "candidate_feature_columns",
            bundle["feature_columns"],
        ),
        "feature_selection": bundle.get(
            "feature_selection",
            {},
        ),
        "sample_count": bundle["sample_count"],
        "realtime_min_packets": bundle.get(
            "realtime_min_packets"
        ),
        "experiment_count": bundle[
            "experiment_count"
        ],
        "stages": {
            "family": _stage_metadata(
                stages["family"]
            ),
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


def _iter_stage_rankings(
    metadata: Mapping[str, Any],
):
    family = metadata["stages"]["family"]
    yield "family", "", family

    for parent, stage in metadata["stages"][
        "architecture_by_family"
    ].items():
        yield "architecture", str(parent), stage

    for parent, stage in metadata["stages"][
        "variant_by_parent"
    ].items():
        yield "variant", str(parent), stage


def _write_fisher_scores_csv(
    metadata: Mapping[str, Any],
    path: Path,
) -> None:
    fields = [
        "mode",
        "feature_mode",
        "window_size_sec",
        "stage",
        "parent",
        "rank",
        "feature",
        "fisher_score",
        "selected",
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

        for stage_name, parent, stage in (
            _iter_stage_rankings(metadata)
        ):
            for item in stage.get(
                "fisher_ranking",
                [],
            ):
                writer.writerow(
                    {
                        "mode": metadata["mode"],
                        "feature_mode": metadata[
                            "feature_mode"
                        ],
                        "window_size_sec": metadata[
                            "window_size_sec"
                        ],
                        "stage": stage_name,
                        "parent": parent,
                        "rank": item.get("rank"),
                        "feature": item.get(
                            "feature"
                        ),
                        "fisher_score": item.get(
                            "fisher_score"
                        ),
                        "selected": bool(
                            item.get(
                                "selected",
                                False,
                            )
                        ),
                    }
                )


def load_bundle(
    path: str | Path,
) -> Dict[str, Any]:
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    if (
        bundle.get("schema_version")
        != MODEL_SCHEMA_VERSION
    ):
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
            "feature_count": 0,
        }

    if kind != "classifier":
        return {
            "label": None,
            "confidence": 0.0,
            "probabilities": {},
            "kind": str(
                kind or "unavailable"
            ),
            "feature_count": 0,
        }

    feature_columns = list(
        stage.get("feature_columns", [])
    )
    if not feature_columns:
        raise ArchitectureModelError(
            "Trained classifier stage has no "
            "stage-specific feature columns"
        )

    vector = np.asarray(
        [
            _numeric(feature_row.get(name))
            for name in feature_columns
        ],
        dtype=np.float64,
    )

    model = stage["model"]
    probabilities = model.predict_proba(
        vector.reshape(1, -1)
    )[0]
    classes = [
        str(value)
        for value in model.classes_
    ]
    distribution = {
        label: float(probability)
        for label, probability in zip(
            classes,
            probabilities,
        )
    }
    label = max(
        distribution,
        key=distribution.get,
    )
    return {
        "label": label,
        "confidence": float(
            distribution[label]
        ),
        "probabilities": distribution,
        "kind": "classifier",
        "feature_count": len(feature_columns),
        "features_used": feature_columns,
    }


def predict_hierarchy(
    bundle: Mapping[str, Any],
    feature_row: Mapping[str, Any],
) -> Dict[str, Any]:
    stages = bundle["stages"]

    family = _predict_stage(
        stages["family"],
        feature_row,
    )
    family_label = family.get("label")

    architecture = {
        "label": None,
        "confidence": 0.0,
        "probabilities": {},
        "kind": "unavailable",
        "feature_count": 0,
    }
    if family_label:
        stage = stages[
            "architecture_by_family"
        ].get(family_label)
        if stage is not None:
            architecture = _predict_stage(
                stage,
                feature_row,
            )

    variant = {
        "label": None,
        "confidence": 0.0,
        "probabilities": {},
        "kind": "unavailable",
        "feature_count": 0,
    }
    architecture_label = architecture.get(
        "label"
    )
    if family_label and architecture_label:
        stage = stages[
            "variant_by_parent"
        ].get(
            f"{family_label}::{architecture_label}"
        )
        if stage is not None:
            variant = _predict_stage(
                stage,
                feature_row,
            )

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
        value = float(
            window_size_sec or 0.0
        )
        token = (
            ("%g" % value)
            .replace(".", "p")
        )
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
    return (
        path
        if path.exists()
        else None
    )
