from __future__ import annotations

"""Memory-safe, leakage-safe packet-budget fingerprint evaluation.

Key properties
--------------
1. The original packet-budget semantics are preserved:
   K=1 is one arriving packet plus its inter-arrival time; K>1 uses exact,
   non-overlapping K-packet blocks produced by the existing dataset builder.
2. Giant packet CSVs are NEVER loaded in full. Each budget file is streamed
   twice: a metadata-only counting pass followed by a deterministic,
   trace-balanced sampling pass.
3. Every experiment/client/source trace remains represented, subject only to
   the configured per-trace cap.
4. Experiment-disjoint OOF evaluation is preserved.
5. Fisher ranking is computed using the TRAINING fold only.
6. Packet-level OOF probabilities are also late-fused per client trace,
   yielding one final client prediction without retraining on held-out data.
7. proxy_to_proxy results are labelled honestly as proxy->proxy, not
   endpoint->proxy.
"""

import argparse
import csv
import hashlib
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai_fingerprint.packet_fingerprinting import METADATA_FIELDS


TARGETS = {
    "family": "family",
    "architecture": "architecture",
    "variant": "variant",
    "application": "application",
}

TRACE_FIELDS = ("experiment_id", "client_id", "source_role")
REQUIRED_FIELDS = {
    "experiment_id",
    "client_id",
    "source_role",
    "family",
    "architecture",
    "variant",
    "application",
}


def slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "all"


def wrap_label(label: str, width: int = 18) -> str:
    display = str(label).replace("_", " ")
    if len(display) <= width:
        return display
    return "\n".join(
        textwrap.wrap(
            display,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def plot_confusion(
    cm: np.ndarray,
    classes: list[str],
    prefix: Path,
    *,
    unit_name: str,
) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(
        cm.astype(float),
        row_sum,
        out=np.zeros_like(cm, dtype=float),
        where=row_sum != 0,
    )
    support = {
        label: int(cm[index].sum())
        for index, label in enumerate(classes)
    }

    size = max(6.0, min(13.0, 4.8 + 0.9 * len(classes)))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    image = ax.imshow(norm, vmin=0.0, vmax=1.0, cmap="Blues")

    xlabels = [wrap_label(x) for x in classes]
    ylabels = [
        f"{wrap_label(x)}\nN={support[x]:,}"
        for x in classes
    ]
    rotation = 0 if len(classes) <= 5 else 35
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(
        xlabels,
        rotation=rotation,
        ha="center" if rotation == 0 else "right",
    )
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel(f"True class ({unit_name})")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{100.0 * norm[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] >= 0.5 else "black",
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Row-normalized proportion")
    fig.tight_layout()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fisher_scores(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
) -> list[tuple[str, float]]:
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        return [(name, 0.0) for name in names]

    overall = np.mean(X, axis=0)
    between = np.zeros(X.shape[1], dtype=float)
    within = np.zeros(X.shape[1], dtype=float)

    for cls in classes:
        mask = y == cls
        Xi = X[mask]
        if len(Xi) == 0:
            continue
        mean = Xi.mean(axis=0)
        between += len(Xi) * (mean - overall) ** 2
        within += ((Xi - mean) ** 2).sum(axis=0)

    score = np.divide(between, within + 1e-12)
    return sorted(
        zip(names, score.tolist()),
        key=lambda x: (-x[1], x[0]),
    )


def source_experiment_weights(
    experiments: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    """Equal total weight per experiment and equal source weight inside it."""
    source_counts = Counter(
        (str(e), str(s))
        for e, s in zip(experiments, sources)
    )
    sources_by_exp: dict[str, set[str]] = defaultdict(set)
    for e, s in zip(experiments, sources):
        sources_by_exp[str(e)].add(str(s))

    weights = []
    for e, s in zip(experiments, sources):
        e = str(e)
        s = str(s)
        weights.append(
            1.0
            / (
                len(sources_by_exp[e])
                * source_counts[(e, s)]
            )
        )
    return np.asarray(weights, dtype=float)


def _stable_seed(
    global_seed: int,
    packet_budget: int,
    trace_key: tuple[str, str, str],
) -> int:
    text = (
        f"{int(global_seed)}|{int(packet_budget)}|"
        + "|".join(str(v) for v in trace_key)
    )
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _stratified_positions(
    n: int,
    cap: int,
    seed: int,
) -> np.ndarray:
    """Select at most cap positions while spanning the entire trace."""
    n = int(n)
    cap = int(cap)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    edges = np.floor(
        np.linspace(0, n, cap + 1, dtype=np.float64)
    ).astype(np.int64)
    positions = np.empty(cap, dtype=np.int64)

    for i in range(cap):
        low = int(edges[i])
        high = int(edges[i + 1])
        if high <= low:
            high = low + 1
        positions[i] = int(rng.integers(low, min(high, n)))

    return np.unique(np.clip(positions, 0, n - 1))


def _read_header(path: Path) -> list[str]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_FIELDS - set(columns)
    if missing:
        raise RuntimeError(
            f"{path}: missing required columns {sorted(missing)}"
        )
    return columns


def count_trace_rows(
    path: Path,
    *,
    chunksize: int,
) -> dict[tuple[str, str, str], int]:
    _read_header(path)
    counts: Counter[tuple[str, str, str]] = Counter()

    for chunk in pd.read_csv(
        path,
        usecols=list(TRACE_FIELDS),
        dtype={
            "experiment_id": "string",
            "client_id": "string",
            "source_role": "string",
        },
        chunksize=int(chunksize),
        low_memory=True,
    ):
        grouped = chunk.groupby(
            list(TRACE_FIELDS),
            sort=False,
            dropna=False,
        ).size()
        for key, value in grouped.items():
            normalized = tuple(str(v) for v in key)
            counts[normalized] += int(value)

    return dict(counts)


def sample_budget_streaming(
    path: Path,
    *,
    packet_budget: int,
    max_samples_per_trace: int,
    chunksize: int,
    sampling_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build a balanced analysis table without loading the full CSV."""
    counts = count_trace_rows(path, chunksize=chunksize)
    if not counts:
        raise RuntimeError(f"{path}: no trace rows found")

    targets: dict[tuple[str, str, str], np.ndarray] = {}
    seeds: dict[tuple[str, str, str], int] = {}
    selected_counts: Counter[tuple[str, str, str]] = Counter()

    for key, n in counts.items():
        seed = _stable_seed(
            sampling_seed,
            packet_budget,
            key,
        )
        seeds[key] = seed
        targets[key] = _stratified_positions(
            n,
            max_samples_per_trace,
            seed,
        )

    seen: Counter[tuple[str, str, str]] = Counter()
    selected_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        path,
        chunksize=int(chunksize),
        low_memory=False,
    ):
        if chunk.empty:
            continue

        for raw_key, group in chunk.groupby(
            list(TRACE_FIELDS),
            sort=False,
            dropna=False,
        ):
            key = tuple(str(v) for v in raw_key)
            if key not in targets:
                continue

            start_seen = int(seen[key])
            end_seen = start_seen + len(group)
            positions = targets[key]

            lo = int(np.searchsorted(
                positions,
                start_seen,
                side="left",
            ))
            hi = int(np.searchsorted(
                positions,
                end_seen,
                side="left",
            ))

            if hi > lo:
                local = (
                    positions[lo:hi]
                    - start_seen
                ).astype(np.int64)
                part = group.iloc[local].copy()
                selected_parts.append(part)
                selected_counts[key] += int(len(part))

            seen[key] = end_seen

    if not selected_parts:
        raise RuntimeError(
            f"{path}: streaming sampler selected no rows"
        )

    sampled = pd.concat(
        selected_parts,
        axis=0,
        ignore_index=True,
    )

    audit: list[dict[str, Any]] = []
    for key in sorted(counts):
        exp, client, role = key
        audit.append({
            "packet_budget": int(packet_budget),
            "experiment_id": exp,
            "client_id": client,
            "source_role": role,
            "available_samples": int(counts[key]),
            "selected_samples": int(selected_counts[key]),
            "max_samples_per_trace": int(max_samples_per_trace),
            "sampling_seed": int(seeds[key]),
            "sampling_method": "deterministic_stratified_across_trace",
        })

    return sampled, audit


def stage_slices(df: pd.DataFrame):
    yield "family", "all", df

    for family, sub in df.groupby("family", sort=True):
        yield "architecture", str(family), sub

    for (family, arch), sub in df.groupby(
        ["family", "architecture"],
        sort=True,
    ):
        yield "variant", f"{family}::{arch}", sub

    for (family, arch, variant), sub in df.groupby(
        ["family", "architecture", "variant"],
        sort=True,
    ):
        yield (
            "application",
            f"{family}::{arch}::{variant}",
            sub,
        )


def class_experiments(
    df: pd.DataFrame,
    target: str,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for label, exp in zip(
        df[target].astype(str),
        df["experiment_id"].astype(str),
    ):
        result[label].add(exp)
    return result


def shared_class_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
) -> dict[str, list[str]]:
    a = class_experiments(train_df, target)
    b = class_experiments(test_df, target)
    result = {}

    for cls in sorted(set(a) & set(b)):
        shared = sorted(a[cls] & b[cls])
        if shared:
            result[cls] = shared

    return result


def make_test_folds(
    shared: dict[str, list[str]],
    n_splits: int,
    seed: int = 42,
) -> list[set[str]]:
    rng = np.random.RandomState(seed)
    folds = [set() for _ in range(n_splits)]

    for cls in sorted(shared):
        exps = list(shared[cls])
        rng.shuffle(exps)
        start = int(rng.randint(0, n_splits))
        for i, exp in enumerate(exps):
            folds[(start + i) % n_splits].add(exp)

    return folds


def predictor_columns(df: pd.DataFrame) -> list[str]:
    ignored = set(METADATA_FIELDS)
    candidates = []

    for name in df.columns:
        if name in ignored or name in TARGETS.values():
            continue
        if pd.api.types.is_numeric_dtype(df[name]):
            candidates.append(name)

    if not candidates:
        raise RuntimeError(
            "No numeric packet predictors remain."
        )
    return candidates


def probability_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    classes: list[str],
) -> dict[str, float]:
    if len(classes) < 2:
        return {}

    yb = np.column_stack([
        (y == c).astype(int)
        for c in classes
    ])
    aucs = []
    aps = []

    for i in range(len(classes)):
        if len(np.unique(yb[:, i])) < 2:
            continue
        try:
            aucs.append(float(
                roc_auc_score(yb[:, i], prob[:, i])
            ))
        except ValueError:
            pass
        try:
            aps.append(float(
                average_precision_score(
                    yb[:, i],
                    prob[:, i],
                )
            ))
        except ValueError:
            pass

    result = {}
    if aucs:
        result["macro_auroc_ovr"] = float(np.mean(aucs))
    if aps:
        result["macro_auprc_ovr"] = float(np.mean(aps))
    return result


def save_metrics_and_matrix(
    out: Path,
    y: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    classes: list[str],
    metadata: dict[str, Any],
    *,
    unit_name: str,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y, pred, labels=classes)

    with (
        out / "confusion_matrix_counts.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *classes])
        for label, row in zip(classes, cm):
            writer.writerow([label, *row.tolist()])

    with (
        out / "confusion_matrix_classifier_samples.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "N", "unit"])
        for index, label in enumerate(classes):
            writer.writerow([
                label,
                int(cm[index].sum()),
                unit_name,
            ])

    plot_confusion(
        cm,
        classes,
        out / "confusion_matrix",
        unit_name=unit_name,
    )

    metrics = {
        **metadata,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(
            precision_score(
                y,
                pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y,
                pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        **probability_metrics(y, prob, classes),
    }

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def save_client_fusion(
    *,
    out: Path,
    pred_rows: list[dict[str, Any]],
    classes: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not pred_rows:
        return {}

    prob_columns = [
        f"probability::{cls}"
        for cls in classes
    ]

    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in pred_rows:
        key = (
            str(row["experiment_id"]),
            str(row["client_id"]),
        )
        groups[key].append(row)

    fused_rows: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []
    probabilities: list[np.ndarray] = []

    for (experiment_id, client_id), rows in sorted(groups.items()):
        truths = {str(row["y_true"]) for row in rows}
        if len(truths) != 1:
            raise RuntimeError(
                "Conflicting labels during packet fusion for "
                f"{experiment_id}/{client_id}: {sorted(truths)}"
            )

        matrix = np.asarray(
            [
                [float(row[column]) for column in prob_columns]
                for row in rows
            ],
            dtype=float,
        )
        mean_prob = matrix.mean(axis=0)
        prediction = classes[int(np.argmax(mean_prob))]
        truth = next(iter(truths))

        item = {
            "experiment_id": experiment_id,
            "client_id": client_id,
            "packet_observation_count": int(len(rows)),
            "y_true": truth,
            "y_pred": prediction,
        }
        for i, cls in enumerate(classes):
            item[f"probability::{cls}"] = float(mean_prob[i])
        fused_rows.append(item)
        y_true.append(truth)
        y_pred.append(prediction)
        probabilities.append(mean_prob)

    fusion_dir = out / "client_trace_fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    with (
        fusion_dir / "oof_fused_predictions.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        fields = list(fused_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fused_rows)

    fusion_meta = {
        **metadata,
        "evaluation_unit": "complete_client_trace_after_packet_probability_fusion",
        "fusion_method": "mean_oof_probability_across_packet_observations",
        "packet_predictions_are_oof": True,
        "client_trace_count": int(len(fused_rows)),
    }

    return save_metrics_and_matrix(
        fusion_dir,
        np.asarray(y_true, dtype=object),
        np.asarray(y_pred, dtype=object),
        np.asarray(probabilities, dtype=float),
        classes,
        fusion_meta,
        unit_name="client traces",
    )


def _evaluation_name(vantage_mode: str) -> str:
    if vantage_mode == "proxy_to_proxy":
        return "proxy_to_proxy_experiment_disjoint_oof"
    return "endpoint_to_proxy_experiment_disjoint_oof"



def experiment_trace_weights(
    experiment_ids: Sequence[str],
    client_ids: Sequence[str],
) -> np.ndarray:
    """Equal experiment -> equal client trace -> equal observation.

    Each independent experiment receives the same total training
    weight. Within an experiment, each logical client trace receives
    an equal share. Within a trace, its weight is divided equally
    across its packet observations.
    """
    experiments = np.asarray(
        experiment_ids,
        dtype=object,
    )

    clients = np.asarray(
        client_ids,
        dtype=object,
    )

    if len(experiments) != len(clients):
        raise RuntimeError(
            "experiment/client arrays have different lengths"
        )

    if len(experiments) == 0:
        return np.asarray([], dtype=float)

    trace_counts: dict[
        tuple[str, str],
        int,
    ] = defaultdict(int)

    experiment_clients: dict[
        str,
        set[str],
    ] = defaultdict(set)

    keys: list[
        tuple[str, str]
    ] = []

    for experiment_id, client_id in zip(
        experiments,
        clients,
    ):
        experiment_id = str(experiment_id)
        client_id = str(client_id)

        key = (
            experiment_id,
            client_id,
        )

        keys.append(key)
        trace_counts[key] += 1
        experiment_clients[
            experiment_id
        ].add(client_id)

    weights = np.zeros(
        len(keys),
        dtype=float,
    )

    for i, (
        experiment_id,
        client_id,
    ) in enumerate(keys):

        observations_in_trace = trace_counts[
            (
                experiment_id,
                client_id,
            )
        ]

        traces_in_experiment = len(
            experiment_clients[
                experiment_id
            ]
        )

        weights[i] = (
            1.0
            / traces_in_experiment
            / observations_in_trace
        )

    mean_weight = float(
        np.mean(weights)
    )

    if (
        not np.isfinite(mean_weight)
        or mean_weight <= 0
    ):
        raise RuntimeError(
            "Invalid packet training weights"
        )

    # Mean = 1, while preserving relative weighting.
    weights /= mean_weight

    return weights


def weighted_fisher_scores(
    X: np.ndarray,
    labels: Sequence[str],
    feature_columns: Sequence[str],
    sample_weight: Sequence[float],
) -> list[tuple[str, float]]:
    """Weighted multiclass Fisher score.

    The weighting matches packet-model fitting:
    equal experiment -> equal client trace -> equal observation.
    """
    if (
        X.ndim != 2
        or X.shape[1] != len(
            feature_columns
        )
    ):
        raise RuntimeError(
            "Invalid matrix for weighted Fisher scoring"
        )

    y = np.asarray(
        labels,
        dtype=object,
    )

    w = np.asarray(
        sample_weight,
        dtype=float,
    )

    if (
        len(w) != X.shape[0]
        or np.any(~np.isfinite(w))
        or np.any(w < 0)
        or float(np.sum(w)) <= 0
    ):
        raise RuntimeError(
            "Invalid Fisher sample weights"
        )

    classes = sorted(
        set(
            str(value)
            for value in labels
        )
    )

    if len(classes) < 2:
        return [
            (name, 0.0)
            for name in feature_columns
        ]

    total_weight = float(
        np.sum(w)
    )

    overall = (
        np.sum(
            X * w[:, None],
            axis=0,
        )
        / total_weight
    )

    numerator = np.zeros(
        X.shape[1],
        dtype=np.float64,
    )

    denominator = np.zeros(
        X.shape[1],
        dtype=np.float64,
    )

    for label in classes:
        mask = y == label

        subset = X[mask]
        class_weights = w[mask]

        class_weight = float(
            np.sum(class_weights)
        )

        if (
            subset.size == 0
            or class_weight <= 0
        ):
            continue

        mean = (
            np.sum(
                subset
                * class_weights[:, None],
                axis=0,
            )
            / class_weight
        )

        variance = (
            np.sum(
                np.square(
                    subset - mean
                )
                * class_weights[:, None],
                axis=0,
            )
            / class_weight
        )

        numerator += (
            class_weight
            * np.square(
                mean - overall
            )
        )

        denominator += (
            class_weight
            * variance
        )

    scores = (
        numerator
        / np.maximum(
            denominator,
            1e-12,
        )
    )

    return sorted(
        [
            (
                name,
                float(score),
            )
            for name, score
            in zip(
                feature_columns,
                scores,
            )
        ],
        key=lambda item:
            item[1],
        reverse=True,
    )


def evaluate_cross_vantage(
    stage_df: pd.DataFrame,
    level: str,
    parent: str,
    train_roles: set[str],
    out: Path,
    *,
    vantage_mode: str,
    packet_budget: int,
    n_estimators: int,
    top_k: int,
    max_folds: int,
    descriptive_variants: bool,
    packet_fusion: bool,
) -> dict[str, Any]:
    target = TARGETS[level]
    train_all = stage_df[
        stage_df["source_role"].isin(train_roles)
    ].copy()
    proxy_all = stage_df[
        stage_df["source_role"] == "proxy"
    ].copy()

    status = {
        "level": level,
        "parent": parent,
        "packet_budget": int(packet_budget),
        "vantage_mode": vantage_mode,
        "train_roles": sorted(train_roles),
        "test_role": "proxy",
        "train_samples_available_after_trace_balancing": int(len(train_all)),
        "proxy_samples_available_after_trace_balancing": int(len(proxy_all)),
    }

    if train_all.empty or proxy_all.empty:
        status.update({
            "status": "missing_vantage_data",
            "reason": "training source or proxy source is empty",
        })
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )
        return status

    shared = shared_class_experiments(train_all, proxy_all, target)
    eligible = {
        cls: exps
        for cls, exps in shared.items()
        if len(exps) >= 2
    }
    status["shared_experiments_per_class"] = {
        k: len(v)
        for k, v in shared.items()
    }
    status["oof_eligible_classes"] = sorted(eligible)
    status["unsupported_classes"] = sorted(
        set(stage_df[target].astype(str)) - set(eligible)
    )

    if len(eligible) >= 2:
        min_groups = min(len(v) for v in eligible.values())
        n_splits = min(max_folds, min_groups)
        valid_experiments = {
            x
            for values in eligible.values()
            for x in values
        }

        train_base = train_all[
            train_all[target].astype(str).isin(eligible)
            & train_all["experiment_id"].astype(str).isin(valid_experiments)
        ]
        proxy_base = proxy_all[
            proxy_all[target].astype(str).isin(eligible)
            & proxy_all["experiment_id"].astype(str).isin(valid_experiments)
        ]

        features = predictor_columns(train_base)
        class_list = sorted(eligible)
        class_idx = {c: i for i, c in enumerate(class_list)}
        folds = make_test_folds(eligible, n_splits)

        all_truth = []
        all_pred = []
        all_prob = []
        pred_rows = []
        fold_details = []

        for fold_no, test_exps in enumerate(folds, start=1):
            tr = train_base[
                ~train_base["experiment_id"].astype(str).isin(test_exps)
            ]
            te = proxy_base[
                proxy_base["experiment_id"].astype(str).isin(test_exps)
            ]

            if tr.empty or te.empty:
                raise RuntimeError(f"Fold {fold_no} is empty")

            train_classes = set(tr[target].astype(str))
            test_classes = set(te[target].astype(str))

            if train_classes != set(class_list) or test_classes != set(class_list):
                raise RuntimeError(
                    f"Invalid fold {fold_no}: "
                    f"train={sorted(train_classes)} "
                    f"test={sorted(test_classes)} "
                    f"expected={class_list}"
                )

            if set(tr["experiment_id"].astype(str)) & set(te["experiment_id"].astype(str)):
                raise RuntimeError(f"Experiment leakage in fold {fold_no}")

            Xtr = tr[features].fillna(0.0).to_numpy(dtype=float)
            ytr = tr[target].astype(str).to_numpy(dtype=object)
            Xte = te[features].fillna(0.0).to_numpy(dtype=float)
            yte = te[target].astype(str).to_numpy(dtype=object)

            weights = experiment_trace_weights(
                tr["experiment_id"].astype(str).to_numpy(),
                tr["client_id"].astype(str).to_numpy(),
            )

            ranked = weighted_fisher_scores(
                Xtr,
                ytr,
                features,
                sample_weight=weights,
            )

            selected = [
                name
                for name, _ in ranked[
                    : min(
                        top_k,
                        len(ranked),
                    )
                ]
            ]

            sel_idx = [
                features.index(name)
                for name in selected
            ]

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42 + fold_no,
                n_jobs=-1,
            )

            model.fit(
                Xtr[:, sel_idx],
                ytr,
                sample_weight=weights,
            )
            pred = model.predict(Xte[:, sel_idx])
            raw = model.predict_proba(Xte[:, sel_idx])

            aligned = np.full(
                (len(te), len(class_list)),
                1e-15,
                dtype=float,
            )
            for j, cls in enumerate(model.classes_):
                aligned[:, class_idx[str(cls)]] = raw[:, j]
            aligned /= aligned.sum(axis=1, keepdims=True)

            all_truth.append(yte)
            all_pred.append(pred)
            all_prob.append(aligned)

            for local_i, (_, row) in enumerate(te.iterrows()):
                item = {
                    "fold": fold_no,
                    "sample_id": row.get("sample_id", ""),
                    "experiment_id": row["experiment_id"],
                    "client_id": row["client_id"],
                    "y_true": yte[local_i],
                    "y_pred": pred[local_i],
                }
                for ci, cls in enumerate(class_list):
                    item[f"probability::{cls}"] = float(aligned[local_i, ci])
                pred_rows.append(item)

            fold_details.append({
                "fold": fold_no,
                "train_samples": int(len(tr)),
                "test_samples": int(len(te)),
                "train_experiments": sorted(set(tr["experiment_id"].astype(str))),
                "test_experiments": sorted(set(te["experiment_id"].astype(str))),
                "train_classes": sorted(train_classes),
                "test_classes": sorted(test_classes),
                "selected_features": selected,
                "experiment_leakage": False,
            })

            del Xtr, Xte, ytr, yte, raw, aligned, model

        y = np.concatenate(all_truth)
        pred = np.concatenate(all_pred)
        prob = np.vstack(all_prob)

        out.mkdir(parents=True, exist_ok=True)

        if pred_rows:
            fields = list(pred_rows[0].keys())
            with (out / "oof_predictions.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(pred_rows)

        (out / "selected_features_by_fold.json").write_text(
            json.dumps(fold_details, indent=2),
            encoding="utf-8",
        )

        status.update({
            "status": "evaluated",
            "evaluation": _evaluation_name(vantage_mode),
            "folds": n_splits,
            "oof_samples": int(len(y)),
            "oof_test_experiments": int(
                len({r["experiment_id"] for r in pred_rows})
            ),
            "fisher_selection": "training_fold_only",
            "training_weight_policy": (
                "equal_experiment_then_equal_client_trace_then_equal_observation"
            ),
            "fisher_weight_policy": (
                "equal_experiment_then_equal_client_trace_then_equal_observation"
            ),
            "sampling_policy": (
                "deterministic_trace_balanced_before_grouped_oof;"
                "label_blind_sampling_uses_only_trace_identity_and_order"
            ),
            "evaluation_unit": f"exact_{packet_budget}_packet_observation",
        })

        result = save_metrics_and_matrix(
            out,
            y,
            pred,
            prob,
            class_list,
            status,
            unit_name=f"{packet_budget}-packet observations",
        )

        if packet_fusion:
            fusion = save_client_fusion(
                out=out,
                pred_rows=pred_rows,
                classes=class_list,
                metadata={
                    **status,
                    "packet_level_accuracy": result["accuracy"],
                    "packet_level_macro_f1": result["macro_f1"],
                },
            )
            if fusion:
                result["client_trace_fusion"] = {
                    key: fusion.get(key)
                    for key in (
                        "accuracy",
                        "balanced_accuracy",
                        "macro_precision",
                        "macro_recall",
                        "macro_f1",
                        "macro_auroc_ovr",
                        "macro_auprc_ovr",
                        "client_trace_count",
                    )
                    if key in fusion
                }

        return result

    if level == "variant" and descriptive_variants:
        common_classes = sorted(
            set(train_all[target].astype(str))
            & set(proxy_all[target].astype(str))
        )

        if len(common_classes) >= 2:
            common_exps = (
                set(train_all["experiment_id"].astype(str))
                & set(proxy_all["experiment_id"].astype(str))
            )
            tr = train_all[
                train_all[target].astype(str).isin(common_classes)
                & train_all["experiment_id"].astype(str).isin(common_exps)
            ]
            te = proxy_all[
                proxy_all[target].astype(str).isin(common_classes)
                & proxy_all["experiment_id"].astype(str).isin(common_exps)
            ]

            if not tr.empty and not te.empty:
                features = predictor_columns(tr)
                Xtr = tr[features].fillna(0.0).to_numpy(dtype=float)
                ytr = tr[target].astype(str).to_numpy(dtype=object)
                Xte = te[features].fillna(0.0).to_numpy(dtype=float)
                yte = te[target].astype(str).to_numpy(dtype=object)

                ranked = fisher_scores(Xtr, ytr, features)
                selected = [
                    n
                    for n, _ in ranked[: min(top_k, len(ranked))]
                ]
                idx = [features.index(n) for n in selected]

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(
                    Xtr[:, idx],
                    ytr,
                    sample_weight=source_experiment_weights(
                        tr["experiment_id"].astype(str).to_numpy(),
                        tr["source_role"].astype(str).to_numpy(),
                    ),
                )

                pred = model.predict(Xte[:, idx])
                raw = model.predict_proba(Xte[:, idx])
                classes = sorted(set(yte.tolist()) | set(ytr.tolist()))
                class_idx = {c: i for i, c in enumerate(classes)}
                prob = np.full(
                    (len(yte), len(classes)),
                    1e-15,
                    dtype=float,
                )
                for j, cls in enumerate(model.classes_):
                    prob[:, class_idx[str(cls)]] = raw[:, j]
                prob /= prob.sum(axis=1, keepdims=True)

                desc = out / "descriptive_same_experiment_transfer"
                status.update({
                    "status": "descriptive_only",
                    "evaluation": "same_experiment_cross_vantage_transfer_not_generalization",
                    "reason": (
                        "insufficient independent experiments per class for "
                        "experiment-disjoint OOF"
                    ),
                    "selected_features": selected,
                    "test_samples": int(len(yte)),
                })

                result = save_metrics_and_matrix(
                    desc,
                    yte,
                    pred,
                    prob,
                    classes,
                    status,
                    unit_name=f"{packet_budget}-packet observations",
                )

                (desc / "DESCRIPTIVE_ONLY.txt").write_text(
                    "Training and proxy observations come from the same coordinated "
                    "experiments. This matrix measures cross-vantage separability "
                    "only; it is not unseen-experiment generalization and must not "
                    "be reported as OOF performance.\n",
                    encoding="utf-8",
                )

                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )
                return result

    status.update({
        "status": "insufficient_independent_runs",
        "reason": (
            "fewer than two shared training/proxy experiment groups for "
            "at least two classes"
        ),
    })
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    return status


def _write_sampling_audit(
    out_root: Path,
    packet_budget: int,
    rows: list[dict[str, Any]],
) -> Path:
    directory = out_root / "_sampling_audit"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"packet_{packet_budget}_sampling_audit.csv"
    fields = [
        "packet_budget",
        "experiment_id",
        "client_id",
        "source_role",
        "available_samples",
        "selected_samples",
        "max_samples_per_trace",
        "sampling_seed",
        "sampling_method",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="fingerprinting_dataset/packet_cross_vantage",
    )
    parser.add_argument(
        "--output",
        default="fingerprinting_results/packet_cross_vantage",
    )
    parser.add_argument(
        "--packet-counts",
        default="1,5,10,25,50,100,250,500",
    )
    parser.add_argument(
        "--vantage-modes",
        default=(
            "client_to_proxy,server_to_proxy,"
            "client_server_to_proxy,proxy_to_proxy"
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument(
        "--max-samples-per-trace",
        type=int,
        default=5000,
        help=(
            "Maximum packet/block observations retained per "
            "experiment/client/source trace for each packet budget."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="CSV streaming chunk size.",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--packet-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--descriptive-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.max_samples_per_trace <= 0:
        raise SystemExit(
            "--max-samples-per-trace must be positive. The full giant CSV "
            "is intentionally not loaded."
        )
    if args.chunksize <= 0:
        raise SystemExit("--chunksize must be positive")

    root = Path(args.dataset_root)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    budgets = [
        int(x)
        for x in args.packet_counts.split(",")
        if x.strip()
    ]
    modes = [
        x.strip()
        for x in args.vantage_modes.split(",")
        if x.strip()
    ]

    mode_roles = {
        "client_to_proxy": {"client"},
        "server_to_proxy": {"server"},
        "client_server_to_proxy": {"client", "server"},
        "proxy_to_proxy": {"proxy"},
    }

    summaries = []
    sampling_summary = []

    for k in budgets:
        path = root / f"packet_{k}.csv"
        if not path.exists():
            print(f"[skip] packet {k}: missing {path}")
            continue

        print(f"\n[stream] {path}")
        print("  counting rows per experiment/client/source ...")
        df, audit = sample_budget_streaming(
            path,
            packet_budget=k,
            max_samples_per_trace=args.max_samples_per_trace,
            chunksize=args.chunksize,
            sampling_seed=args.sampling_seed,
        )

        audit_path = _write_sampling_audit(out_root, k, audit)
        sampling_summary.append({
            "packet_budget": k,
            "available_samples": int(
                sum(row["available_samples"] for row in audit)
            ),
            "selected_samples": int(len(df)),
            "trace_count": int(len(audit)),
            "audit_csv": str(audit_path),
        })

        print(
            f"  balanced samples: {len(df):,} "
            f"across {len(audit):,} traces"
        )
        print(f"  audit: {audit_path}")

        for mode in modes:
            if mode not in mode_roles:
                raise SystemExit(f"Unknown vantage mode: {mode}")
            roles = mode_roles[mode]

            for level, parent, stage_df in stage_slices(df):
                destination = (
                    out_root
                    / mode
                    / f"packet_{k}"
                    / level
                    / slug(parent)
                )
                print(
                    f"[evaluate] k={k} {mode} {level} {parent} "
                    f"samples={len(stage_df):,}"
                )
                result = evaluate_cross_vantage(
                    stage_df,
                    level,
                    parent,
                    roles,
                    destination,
                    vantage_mode=mode,
                    packet_budget=k,
                    n_estimators=args.n_estimators,
                    top_k=args.top_k,
                    max_folds=args.max_folds,
                    descriptive_variants=args.descriptive_variants,
                    packet_fusion=args.packet_fusion,
                )
                summaries.append({
                    "packet_budget": k,
                    "vantage_mode": mode,
                    **result,
                })

        del df

    if summaries:
        fields = sorted(set().union(*(row.keys() for row in summaries)))
        with (out_root / "evaluation_summary.csv").open(
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
            for row in summaries:
                normalized = {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
                writer.writerow(normalized)

    run_summary = {
        "packet_counts": budgets,
        "vantage_modes": modes,
        "split_policy": "experiment_disjoint",
        "training_source": "endpoint_or_proxy_as_selected",
        "evaluation_source": "proxy",
        "single_packet_semantics": "one arriving packet plus its inter-arrival time",
        "fixed_packet_semantics": "non-overlapping exact-K packet blocks",
        "sampling": {
            "policy": "deterministic_stratified_per_experiment_client_source_trace",
            "max_samples_per_trace": int(args.max_samples_per_trace),
            "sampling_seed": int(args.sampling_seed),
            "chunksize": int(args.chunksize),
            "sampling_is_label_blind": True,
            "sampling_summary": sampling_summary,
        },
        "fisher_selection": "inside_training_fold_only",
        "packet_fusion": bool(args.packet_fusion),
        "important_note": (
            "Packet/block observations are nested within client traces and "
            "experiments. They are prediction units, not independent "
            "experimental replications."
        ),
    }

    (out_root / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
