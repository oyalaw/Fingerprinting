from __future__ import annotations

"""Leakage-safe hierarchical fingerprint evaluation across five representations.

Representations evaluated in one run:
  * 0.5 s windows
  * 1 s windows
  * 2 s windows
  * 5 s windows
  * aggregated/overall client traces

The evaluator preserves experiment-level grouping.  All rows from the same
experiment stay on the same side of a fold.  Fisher feature selection is
performed inside each training fold only.

Classes with fewer than two independent experiment groups cannot be evaluated
with group-disjoint OOF cross-validation.  They are reported in coverage files
and excluded from inferential OOF metrics rather than blocking the other
classes.  Optional descriptive in-sample confusion matrices are explicitly
marked DESCRIPTIVE_ONLY.
"""

import argparse
import csv
import json
import math
import textwrap
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ai_fingerprint.architecture_models import (
    FISHER_TOP_K,
    _joined_rows,
    _matrix,
    candidate_feature_columns,
    fisher_score_ranking,
    read_xy,
)


REPRESENTATIONS = [
    ("realtime_0.5s", "window", 0.5, "0.5 s"),
    ("realtime_1s", "window", 1.0, "1 s"),
    ("realtime_2s", "window", 2.0, "2 s"),
    ("realtime_5s", "window", 5.0, "5 s"),
    ("aggregated", "overall", None, "Aggregated"),
]


def slug(value: Any) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "all"


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def write_matrix_csv(path: Path, matrix: np.ndarray, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *classes])
        for label, row in zip(classes, matrix):
            writer.writerow([label, *row.tolist()])


def wrap_class_label(label: str, width: int = 18) -> str:
    """Wrap long class labels for compact confusion-matrix axes.

    Underscores are rendered as spaces so labels such as
    ``convolutional_autoencoder`` can wrap naturally across lines.
    The underlying class value is unchanged; this only affects display.
    """
    display = str(label).replace("_", " ").strip()
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



def _compact_count(value: int) -> str:
    """Format a large integer compactly for figure annotations."""
    value = int(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def _aggregated_packet_annotation(
    rows: list[dict[str, Any]],
) -> str | None:
    """Return underlying packet count for an aggregated/overall matrix.

    The returned n is the number of proxy-observed packet observations
    summarized by the aggregate fingerprints represented in the matrix.
    It is NOT the number of independent classifier samples.
    """
    if not rows:
        return None

    row_types = {
        str(row.get("row_type", "")).strip().lower()
        for row in rows
    }

    # Only annotate complete-trace aggregate representations.
    if row_types != {"overall"}:
        return None

    total = 0
    found = False

    for row in rows:
        raw = row.get("packet_count_total")

        if raw in (None, ""):
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(value) or value < 0:
            continue

        total += int(round(value))
        found = True

    if not found:
        return None

    return f"n = {_compact_count(total)} packets"



def _confusion_sample_annotation(
    rows: list[dict[str, Any]],
) -> str | None:
    """Return the appropriate n annotation for a confusion matrix."""
    if not rows:
        return None

    row_types = {
        str(row.get("row_type", "")).strip().lower()
        for row in rows
    }

    if row_types == {"overall"}:
        return _aggregated_packet_annotation(rows)

    if row_types == {"window"}:
        return f"n = {_compact_count(len(rows))} windows"

    return None


def plot_confusion_counts_and_percent(
    matrix: np.ndarray,
    classes: list[str],
    *,
    path_prefix: Path,
    sample_annotation: str | None = None,
) -> None:
    """Plot a title-free confusion matrix with wrapped class labels.

    Cells show raw classification counts and row-normalized percentages.
    Figure titles/subtitles are intentionally omitted so the plot can be
    captioned cleanly in papers and composite figures.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(float),
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    size = max(6.0, min(13.0, 4.8 + 0.9 * len(classes)))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")

    wrapped_classes = [wrap_class_label(label) for label in classes]
    rotation = 0 if len(classes) <= 5 else 35
    horizontal_alignment = "center" if rotation == 0 else "right"
    tick_fontsize = 9 if len(classes) <= 6 else 8

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(
        wrapped_classes,
        rotation=rotation,
        ha=horizontal_alignment,
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(wrapped_classes, fontsize=tick_fontsize)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    if sample_annotation:
        ax.set_title(
            sample_annotation,
            fontsize=11,
            pad=12,
        )

    for i in range(len(classes)):
        for j in range(len(classes)):
            count = int(matrix[i, j])
            pct = float(normalized[i, j]) * 100.0
            cell_text = f"{pct:.1f}%"
            ax.text(
                j,
                i,
                cell_text,
                ha="center",
                va="center",
                fontsize=8 if len(classes) <= 6 else 7,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Row-normalized proportion")
    fig.tight_layout()
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_coverage_status(
    counts: dict[str, int],
    eligible: set[str],
    path_prefix: Path,
) -> None:
    classes = sorted(counts)
    values = [counts[c] for c in classes]
    labels = ["OOF eligible" if c in eligible else "descriptive only" for c in classes]

    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 1.1), 5.5))
    bars = ax.bar(range(len(classes)), values)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Independent experiment groups")
    ax.set_title("Independent experiment coverage by class")
    ax.axhline(2, linestyle="--", linewidth=1)

    for bar, value, label in zip(bars, values, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value}\n{label}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def group_counts(y: Iterable[str], groups: Iterable[str]) -> dict[str, int]:
    result: dict[str, set[str]] = {}
    for label, group in zip(y, groups):
        result.setdefault(str(label), set()).add(str(group))
    return {label: len(values) for label, values in sorted(result.items())}


def experiment_stratified_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    *,
    random_state: int = 42,
):
    """Create deterministic class-balanced folds at the experiment level.

    Each experiment must map to exactly one class at the current hierarchy stage.
    Experiments for every class are shuffled and distributed round-robin across
    test folds.  This guarantees that every fold contains every evaluated class
    when n_splits <= the minimum independent experiment count per class.
    """
    group_to_labels: dict[str, set[str]] = defaultdict(set)
    for label, group in zip(y.tolist(), groups.tolist()):
        group_to_labels[str(group)].add(str(label))

    bad = {g: sorted(v) for g, v in group_to_labels.items() if len(v) != 1}
    if bad:
        raise RuntimeError(
            "Experiment groups map to multiple labels at one hierarchy stage: "
            + json.dumps(bad, sort_keys=True)
        )

    group_to_label = {g: next(iter(v)) for g, v in group_to_labels.items()}
    by_class: dict[str, list[str]] = defaultdict(list)
    for group, label in sorted(group_to_label.items()):
        by_class[label].append(group)

    rng = np.random.RandomState(random_state)
    fold_test_groups = [set() for _ in range(n_splits)]
    for label in sorted(by_class):
        class_groups = list(by_class[label])
        rng.shuffle(class_groups)
        # Rotate the starting fold between classes to avoid systematic size bias.
        start = int(rng.randint(0, n_splits))
        for i, group in enumerate(class_groups):
            fold_test_groups[(start + i) % n_splits].add(group)

    all_groups = set(group_to_label)
    all_classes = set(by_class)
    for fold_idx, test_groups in enumerate(fold_test_groups, start=1):
        train_groups = all_groups - test_groups
        test_mask = np.asarray([str(g) in test_groups for g in groups], dtype=bool)
        train_mask = ~test_mask
        train_index = np.flatnonzero(train_mask)
        test_index = np.flatnonzero(test_mask)

        train_classes = set(y[train_index].tolist())
        test_classes = set(y[test_index].tolist())
        if train_classes != all_classes or test_classes != all_classes:
            raise RuntimeError(
                f"Invalid experiment-level fold {fold_idx}: "
                f"train_classes={sorted(train_classes)} "
                f"test_classes={sorted(test_classes)} "
                f"expected={sorted(all_classes)}"
            )
        if set(groups[train_index].tolist()) & set(groups[test_index].tolist()):
            raise RuntimeError(f"Experiment leakage detected in fold {fold_idx}.")

        yield train_index, test_index


def experiment_equal_sample_weights(groups: np.ndarray) -> np.ndarray:
    """Weight rows so each experiment contributes equal total training weight."""
    counts = Counter(str(g) for g in groups.tolist())
    return np.asarray([1.0 / counts[str(g)] for g in groups.tolist()], dtype=float)


def experiment_balanced_accuracy(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> float:
    values = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if np.any(mask):
            values.append(float(accuracy_score(truth[mask], prediction[mask])))
    return float(np.mean(values)) if values else float("nan")


def stage_definitions(rows: list[dict[str, Any]]):
    yield "family", "all", rows, "_family"

    families = sorted({str(row["_family"]) for row in rows})
    for family in families:
        subset = [row for row in rows if str(row["_family"]) == family]
        yield "architecture", family, subset, "_architecture"

    parents = sorted(
        {
            (str(row["_family"]), str(row["_architecture"]))
            for row in rows
        }
    )
    for family, architecture in parents:
        subset = [
            row
            for row in rows
            if str(row["_family"]) == family
            and str(row["_architecture"]) == architecture
        ]
        yield "variant", f"{family}::{architecture}", subset, "_variant"

    app_parents = sorted(
        {
            (
                str(row["_family"]),
                str(row["_architecture"]),
                str(row["_variant"]),
            )
            for row in rows
        }
    )
    for family, architecture, variant in app_parents:
        subset = [
            row
            for row in rows
            if str(row["_family"]) == family
            and str(row["_architecture"]) == architecture
            and str(row["_variant"]) == variant
        ]
        yield (
            "application",
            f"{family}::{architecture}::{variant}",
            subset,
            "_application",
        )


def write_coverage_status(
    output_dir: Path,
    counts: dict[str, int],
    eligible: set[str],
) -> tuple[str, str, str]:
    csv_path = output_dir / "coverage_status.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class",
                "independent_experiments",
                "oof_eligible",
                "reason",
            ],
        )
        writer.writeheader()
        for label in sorted(counts):
            ok = label in eligible
            writer.writerow(
                {
                    "class": label,
                    "independent_experiments": counts[label],
                    "oof_eligible": ok,
                    "reason": (
                        "at least 2 independent experiments"
                        if ok
                        else "fewer than 2 independent experiments"
                    ),
                }
            )

    prefix = output_dir / "coverage_status"
    plot_coverage_status(counts, eligible, prefix)
    return str(csv_path), str(prefix.with_suffix(".png")), str(prefix.with_suffix(".pdf"))


def save_oof_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    folds: np.ndarray,
    classes: list[str],
) -> None:
    fields = [
        "row_id",
        "experiment_id",
        "client_capture_id",
        "resolved_client_id",
        "row_type",
        "window_size_sec",
        "window_index",
        "fold",
        "y_true",
        "y_pred",
    ]
    prob_fields = [f"probability::{label}" for label in classes]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + prob_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            output = {
                "row_id": row.get("row_id", ""),
                "experiment_id": row.get("experiment_id", ""),
                "client_capture_id": row.get("client_capture_id", ""),
                "resolved_client_id": row.get("resolved_client_id", ""),
                "row_type": row.get("row_type", ""),
                "window_size_sec": row.get("window_size_sec", ""),
                "window_index": row.get("window_index", ""),
                "fold": int(folds[i]),
                "y_true": truth[i],
                "y_pred": prediction[i],
            }
            for j, label in enumerate(classes):
                output[f"probability::{label}"] = float(probability[i, j])
            writer.writerow(output)


def save_fold_sizes(output_dir: Path, fold_details: list[dict[str, Any]]) -> str:
    path = output_dir / "fold_sizes.csv"
    fields = [
        "fold",
        "train_samples",
        "test_samples",
        "train_experiments",
        "test_experiments",
        "train_classes",
        "test_classes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in fold_details:
            writer.writerow(
                {
                    "fold": item["fold"],
                    "train_samples": item["train_samples"],
                    "test_samples": item["test_samples"],
                    "train_experiments": item["train_experiments"],
                    "test_experiments": item["test_experiments"],
                    "train_classes": "|".join(item["train_classes"]),
                    "test_classes": "|".join(item["test_classes"]),
                }
            )
    return str(path)


def descriptive_confusion(
    rows: list[dict[str, Any]],
    candidate_columns: list[str],
    target_key: str,
    output_dir: Path,
    *,
    n_estimators: int,
    top_k: int,
    representation_label: str,
    stage_label: str,
) -> dict[str, Any]:
    """In-sample descriptive result. Never present as OOF generalization."""
    y = np.asarray([str(row[target_key]) for row in rows], dtype=object)
    classes = sorted(set(y))
    if len(classes) < 2 or not rows:
        return {
            "descriptive_confusion_available": False,
            "descriptive_confusion_reason": "Fewer than two classes are available.",
        }

    X = _matrix(rows, candidate_columns)
    ranking = fisher_score_ranking(X, y, candidate_columns)
    selected = [item["feature"] for item in ranking[: min(top_k, len(ranking))]]
    col_index = {name: i for i, name in enumerate(candidate_columns)}
    selected_idx = [col_index[name] for name in selected]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X[:, selected_idx], y)
    pred = model.predict(X[:, selected_idx])
    cm = confusion_matrix(y, pred, labels=classes)

    desc_dir = output_dir / "descriptive_only"
    desc_dir.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(desc_dir / "confusion_matrix_counts.csv", cm, classes)
    plot_confusion_counts_and_percent(
        cm,
        classes,
        path_prefix=desc_dir / "confusion_matrix",
        sample_annotation=_confusion_sample_annotation(rows),
    )
    warning = desc_dir / "DESCRIPTIVE_ONLY.txt"
    warning.write_text(
        "This result is descriptive/in-sample only. At least one class has "
        "fewer than two independent experiment groups, so experiment-disjoint "
        "out-of-fold evaluation is not possible. Do not report this confusion "
        "matrix as generalization performance.\n",
        encoding="utf-8",
    )

    return {
        "descriptive_confusion_available": True,
        "descriptive_confusion_matrix": str(desc_dir / "confusion_matrix.pdf"),
        "descriptive_warning": str(warning),
        "descriptive_selected_features": selected,
    }


def evaluate_stage(
    rows: list[dict[str, Any]],
    candidate_columns: list[str],
    target_key: str,
    output_dir: Path,
    *,
    representation_label: str,
    stage_level: str,
    parent: str,
    n_estimators: int,
    top_k: int,
    max_folds: int,
    descriptive_policy: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    y_all = np.asarray([str(row[target_key]) for row in rows], dtype=object)
    groups_all = np.asarray(
        [str(row.get("experiment_id", "")) for row in rows], dtype=object
    )
    classes_all = sorted(set(y_all))
    experiment_counts = group_counts(y_all.tolist(), groups_all.tolist())
    eligible_classes = {label for label, count in experiment_counts.items() if count >= 2}
    unsupported_classes = sorted(set(classes_all) - eligible_classes)

    coverage_csv, coverage_png, coverage_pdf = write_coverage_status(
        output_dir, experiment_counts, eligible_classes
    )

    status: dict[str, Any] = {
        "sample_count": len(rows),
        "experiment_count": len(set(groups_all)),
        "classes": classes_all,
        "independent_groups_per_class": experiment_counts,
        "oof_eligible_classes": sorted(eligible_classes),
        "oof_unsupported_classes": unsupported_classes,
        "coverage_status_csv": coverage_csv,
        "coverage_status_png": coverage_png,
        "coverage_status_pdf": coverage_pdf,
        "representation": representation_label,
        "level": stage_level,
        "parent": parent,
    }

    if len(classes_all) < 2:
        status.update({
            "status": "constant",
            "reason": "Only one class is available.",
        })
        (output_dir / "metrics.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        return status

    # Preserve valid OOF evaluation for well-covered classes instead of allowing
    # one newly-added single-run class (e.g. a first Transformer experiment) to
    # invalidate all established classes.
    eligible_mask = np.asarray([label in eligible_classes for label in y_all])
    eligible_rows = [row for row, keep in zip(rows, eligible_mask) if keep]
    y = y_all[eligible_mask]
    groups = groups_all[eligible_mask]
    classes = sorted(set(y))

    if len(classes) < 2:
        status.update({
            "status": "insufficient_independent_runs",
            "reason": (
                "Fewer than two classes have at least two independent "
                "experiment groups; group-disjoint OOF evaluation is not possible."
            ),
        })
        should_describe = (
            descriptive_policy == "all"
            or (descriptive_policy == "final" and representation_label == "Aggregated")
            or (descriptive_policy == "variants" and stage_level == "variant")
        )
        if should_describe:
            status.update(
                descriptive_confusion(
                    rows,
                    candidate_columns,
                    target_key,
                    output_dir,
                    n_estimators=n_estimators,
                    top_k=top_k,
                    representation_label=representation_label,
                    stage_label=f"{stage_level}: {parent}",
                )
            )
        (output_dir / "metrics.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        return status

    eligible_counts = group_counts(y.tolist(), groups.tolist())
    minimum_groups = min(eligible_counts.values())
    n_splits = min(max_folds, minimum_groups)
    if n_splits < 2:
        status.update({
            "status": "insufficient_independent_runs",
            "reason": "At least two independent groups per evaluated class are required.",
        })
        (output_dir / "metrics.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        return status

    X = _matrix(eligible_rows, candidate_columns)
    column_index = {name: i for i, name in enumerate(candidate_columns)}
    class_index = {label: i for i, label in enumerate(classes)}

    oof_prediction = np.empty(len(eligible_rows), dtype=object)
    oof_probability = np.full(
        (len(eligible_rows), len(classes)), np.nan, dtype=np.float64
    )
    oof_fold = np.full(len(eligible_rows), -1, dtype=int)
    fold_details: list[dict[str, Any]] = []

    split_iter = experiment_stratified_splits(
        y, groups, n_splits, random_state=42
    )
    for fold, (train_index, test_index) in enumerate(split_iter, start=1):
        fisher = fisher_score_ranking(
            X[train_index], y[train_index], candidate_columns
        )
        selected = [
            item["feature"]
            for item in fisher[: min(top_k, len(fisher))]
        ]
        selected_index = [column_index[name] for name in selected]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42 + fold,
            n_jobs=-1,
        )
        train_weights = experiment_equal_sample_weights(groups[train_index])
        model.fit(
            X[np.ix_(train_index, selected_index)],
            y[train_index],
            sample_weight=train_weights,
        )
        pred = model.predict(X[np.ix_(test_index, selected_index)])
        raw_prob = model.predict_proba(X[np.ix_(test_index, selected_index)])

        aligned = np.full((len(test_index), len(classes)), 1e-15, dtype=np.float64)
        for source_index, label in enumerate(model.classes_):
            aligned[:, class_index[str(label)]] = raw_prob[:, source_index]
        aligned /= aligned.sum(axis=1, keepdims=True)

        oof_prediction[test_index] = pred
        oof_probability[test_index] = aligned
        oof_fold[test_index] = fold

        fold_details.append(
            {
                "fold": fold,
                "train_samples": int(len(train_index)),
                "test_samples": int(len(test_index)),
                "train_experiments": int(len(set(groups[train_index]))),
                "test_experiments": int(len(set(groups[test_index]))),
                "train_classes": sorted(set(y[train_index].tolist())),
                "test_classes": sorted(set(y[test_index].tolist())),
                "selected_features": selected,
                "class_coverage_valid": True,
                "experiment_leakage": False,
                "training_weight_policy": "equal_total_weight_per_experiment",
            }
        )

    if np.any(oof_fold < 0):
        raise RuntimeError("Some eligible observations did not receive an OOF prediction.")

    pooled: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, oof_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_prediction)),
        "macro_precision": float(
            precision_score(y, oof_prediction, labels=classes, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y, oof_prediction, labels=classes, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y, oof_prediction, labels=classes, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y, oof_prediction, labels=classes, average="micro", zero_division=0)
        ),
        "log_loss": float(log_loss(y, oof_probability, labels=classes)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y, oof_prediction)),
        "cohen_kappa": float(cohen_kappa_score(y, oof_prediction)),
        "experiment_macro_accuracy": experiment_balanced_accuracy(
            y, oof_prediction, groups
        ),
    }

    cm = confusion_matrix(y, oof_prediction, labels=classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    write_matrix_csv(output_dir / "confusion_matrix_counts.csv", cm, classes)
    write_matrix_csv(output_dir / "confusion_matrix_normalized.csv", cm_normalized, classes)
    plot_confusion_counts_and_percent(
        cm,
        classes,
        path_prefix=output_dir / "confusion_matrix",
        sample_annotation=_confusion_sample_annotation(eligible_rows),
    )

    report = classification_report(
        y,
        oof_prediction,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    total = cm.sum()
    with (output_dir / "per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class", "precision", "recall", "f1", "support", "specificity"],
        )
        writer.writeheader()
        for index, label in enumerate(classes):
            tp = cm[index, index]
            fn = cm[index, :].sum() - tp
            fp = cm[:, index].sum() - tp
            tn = total - tp - fn - fp
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            item = report[label]
            writer.writerow(
                {
                    "class": label,
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1-score"],
                    "support": item["support"],
                    "specificity": specificity,
                }
            )

    # AUROC / AUPRC only where mathematically defined.
    y_binary = np.column_stack([(y == label).astype(int) for label in classes])
    roc_results: list[dict[str, Any]] = []
    pr_results: list[dict[str, Any]] = []

    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

    for index, label in enumerate(classes):
        target_binary = y_binary[:, index]
        score = oof_probability[:, index]
        if len(set(target_binary.tolist())) < 2:
            continue

        fpr, tpr, _ = roc_curve(target_binary, score)
        class_auc = roc_auc_score(target_binary, score)
        precision, recall, _ = precision_recall_curve(target_binary, score)
        class_auprc = average_precision_score(target_binary, score)

        roc_results.append({"class": label, "auroc": float(class_auc)})
        pr_results.append({"class": label, "auprc": float(class_auprc)})
        ax_roc.plot(fpr, tpr, label=f"{label} (AUC={class_auc:.3f})")
        ax_pr.plot(recall, precision, label=f"{label} (AP={class_auprc:.3f})")

    if roc_results:
        pooled["macro_auroc_ovr"] = float(np.mean([x["auroc"] for x in roc_results]))
        pooled["micro_auroc_ovr"] = float(
            roc_auc_score(y_binary.ravel(), oof_probability.ravel())
        )
        ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title(f"One-vs-rest ROC — {representation_label}")
        ax_roc.legend(fontsize=8)
        ax_roc.grid(alpha=0.3)
        fig_roc.tight_layout()
        fig_roc.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        fig_roc.savefig(output_dir / "roc_curve.pdf", bbox_inches="tight")

    if pr_results:
        pooled["macro_auprc_ovr"] = float(np.mean([x["auprc"] for x in pr_results]))
        pooled["micro_auprc_ovr"] = float(
            average_precision_score(y_binary.ravel(), oof_probability.ravel())
        )
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"One-vs-rest Precision-Recall — {representation_label}")
        ax_pr.legend(fontsize=8)
        ax_pr.grid(alpha=0.3)
        fig_pr.tight_layout()
        fig_pr.savefig(output_dir / "pr_curve.png", dpi=300, bbox_inches="tight")
        fig_pr.savefig(output_dir / "pr_curve.pdf", bbox_inches="tight")

    plt.close(fig_roc)
    plt.close(fig_pr)

    with (output_dir / "roc_auc_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "auroc"])
        writer.writeheader()
        writer.writerows(roc_results)

    with (output_dir / "pr_auc_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "auprc"])
        writer.writeheader()
        writer.writerows(pr_results)

    save_oof_predictions(
        output_dir / "oof_predictions.csv",
        eligible_rows,
        y,
        oof_prediction,
        oof_probability,
        oof_fold,
        classes,
    )
    (output_dir / "selected_features_by_fold.json").write_text(
        json.dumps(fold_details, indent=2), encoding="utf-8"
    )
    fold_sizes_csv = save_fold_sizes(output_dir, fold_details)

    status.update(
        {
            "status": "evaluated",
            "folds": n_splits,
            "evaluation": "experiment_stratified_group_disjoint_out_of_fold",
            "split_policy": "class-balanced deterministic experiment-level folds",
            "training_weight_policy": "equal_total_weight_per_experiment + balanced_subsample class weighting",
            "fisher_selection": "training_fold_only",
            "oof_sample_count": len(eligible_rows),
            "oof_experiment_count": len(set(groups)),
            "fold_sizes_csv": fold_sizes_csv,
            **pooled,
        }
    )

    # If unsupported classes exist, optionally provide descriptive context while
    # keeping inferential metrics limited to eligible classes.
    should_describe = (
        descriptive_policy == "all"
        or (descriptive_policy == "final" and representation_label == "Aggregated")
        or (descriptive_policy == "variants" and stage_level == "variant")
    )
    if unsupported_classes and should_describe:
        status.update(
            descriptive_confusion(
                rows,
                candidate_columns,
                target_key,
                output_dir,
                n_estimators=n_estimators,
                top_k=top_k,
                representation_label=representation_label,
                stage_label=f"{stage_level}: {parent}",
            )
        )

    (output_dir / "metrics.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return status


def parse_windows(value: str) -> list[float]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(float(item))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x",
        default="fingerprinting_dataset/fingerprinting_X_proxy.csv",
    )
    parser.add_argument(
        "--y",
        default="fingerprinting_dataset/fingerprinting_Y_ground_truth.csv",
    )
    parser.add_argument(
        "--output",
        default="fingerprinting_results/multi_representation",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "final", "realtime"],
        default="all",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["both", "full", "size_normalized"],
        default="size_normalized",
    )
    parser.add_argument("--windows", default="0.5,1,2,5")
    parser.add_argument("--top-k", type=int, default=FISHER_TOP_K)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument(
        "--descriptive-fallback",
        choices=["none", "final", "variants", "all"],
        default="final",
        help=("Generate clearly-marked in-sample confusion matrices when OOF is impossible. "
              "Use variants to emit descriptive variant matrices for every representation."),
    )
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    x_rows, y_by_id = read_xy(args.x, args.y)
    feature_modes = (
        ["full", "size_normalized"]
        if args.feature_mode == "both"
        else [args.feature_mode]
    )

    requested_windows = parse_windows(args.windows)
    jobs: list[tuple[str, str, float | None, str]] = []
    if args.mode in {"all", "realtime"}:
        for window in requested_windows:
            jobs.append((f"realtime_{window:g}s", "window", window, f"{window:g} s"))
    if args.mode in {"all", "final"}:
        jobs.append(("aggregated", "overall", None, "Aggregated"))

    summaries: list[dict[str, Any]] = []
    representation_inventory: list[dict[str, Any]] = []

    for job_name, row_type, window, rep_label in jobs:
        joined = _joined_rows(
            x_rows,
            y_by_id,
            row_type=row_type,
            window_size_sec=window,
        )
        if not joined:
            print(f"[skip] {rep_label}: no matching rows")
            continue

        representation_inventory.append(
            {
                "representation": rep_label,
                "row_type": row_type,
                "window_size_sec": "" if window is None else window,
                "sample_count": len(joined),
                "experiment_count": len({str(r.get("experiment_id", "")) for r in joined}),
                "client_trace_count": len(
                    {
                        (
                            str(r.get("experiment_id", "")),
                            str(r.get("resolved_client_id") or r.get("client_capture_id") or ""),
                        )
                        for r in joined
                    }
                ),
            }
        )

        for feature_mode in feature_modes:
            fields = [key for key in joined[0].keys() if not key.startswith("_")]
            candidate_columns = candidate_feature_columns(fields, feature_mode)
            if not candidate_columns:
                raise RuntimeError(
                    f"No candidate features for feature_mode={feature_mode} representation={rep_label}"
                )

            for level, parent, stage_rows, target in stage_definitions(joined):
                path = output_root / feature_mode / job_name / level / slug(parent)
                print(
                    f"[evaluate] feature_mode={feature_mode} representation={rep_label} "
                    f"level={level} parent={parent} samples={len(stage_rows):,}"
                )
                result = evaluate_stage(
                    stage_rows,
                    candidate_columns,
                    target,
                    path,
                    representation_label=rep_label,
                    stage_level=level,
                    parent=parent,
                    n_estimators=args.n_estimators,
                    top_k=args.top_k,
                    max_folds=args.max_folds,
                    descriptive_policy=args.descriptive_fallback,
                )
                summaries.append(
                    {
                        "feature_mode": feature_mode,
                        "representation": rep_label,
                        "mode": "aggregated" if row_type == "overall" else "realtime",
                        "window_size_sec": "" if window is None else window,
                        "level": level,
                        "parent": parent,
                        **result,
                    }
                )

    summary_fields = [
        "feature_mode",
        "representation",
        "mode",
        "window_size_sec",
        "level",
        "parent",
        "status",
        "sample_count",
        "experiment_count",
        "oof_sample_count",
        "oof_experiment_count",
        "folds",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_f1",
        "log_loss",
        "matthews_correlation_coefficient",
        "cohen_kappa",
        "experiment_macro_accuracy",
        "macro_auroc_ovr",
        "micro_auroc_ovr",
        "macro_auprc_ovr",
        "micro_auprc_ovr",
        "fold_sizes_csv",
        "coverage_status_csv",
        "descriptive_confusion_available",
        "descriptive_confusion_matrix",
        "descriptive_warning",
    ]
    summary_path = output_root / "evaluation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    inventory_path = output_root / "representation_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "representation",
                "row_type",
                "window_size_sec",
                "sample_count",
                "experiment_count",
                "client_trace_count",
            ],
        )
        writer.writeheader()
        writer.writerows(representation_inventory)

    (output_root / "evaluation_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    print("\nEvaluation complete:")
    print(f"  summary:   {summary_path}")
    print(f"  inventory: {inventory_path}")


if __name__ == "__main__":
    main()
