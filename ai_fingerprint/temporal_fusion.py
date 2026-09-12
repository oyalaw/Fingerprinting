from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .architecture_models import FISHER_TOP_K, REALTIME_MIN_PACKETS, candidate_feature_columns, fisher_score_ranking
from .round_fingerprinting import (
    RoundFingerprintingError,
    _display_name,
    _matrix,
    _metrics_from_predictions,
    _require_ml,
    _safe_slug,
    _write_confusion_artifacts,
    _write_roc_pr,
)


def _numeric(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _load_window_rows(
    x_csv: str | Path,
    y_csv: str | Path,
    *,
    window_size_sec: float,
) -> list[Dict[str, Any]]:
    with Path(x_csv).open(newline="", encoding="utf-8") as handle:
        x_rows = list(csv.DictReader(handle))
    with Path(y_csv).open(newline="", encoding="utf-8") as handle:
        y_rows = list(csv.DictReader(handle))
    y_by_id = {str(row.get("row_id") or ""): row for row in y_rows}

    joined = []
    for x in x_rows:
        if str(x.get("row_type") or "") != "window":
            continue
        packets = _numeric(x.get("packet_count_total"))
        if packets < REALTIME_MIN_PACKETS:
            continue
        current = _numeric(x.get("window_size_sec"))
        if current <= 0:
            current = max(
                0.0,
                _numeric(x.get("window_end_sec")) - _numeric(x.get("window_start_sec")),
            )
        if not math.isclose(current, float(window_size_sec), rel_tol=0.0, abs_tol=1e-9):
            continue
        y = y_by_id.get(str(x.get("row_id") or ""))
        if not y:
            continue
        row = dict(x)
        row["_y"] = y
        joined.append(row)
    return joined


def _groups_per_class(rows, target: str) -> Dict[str, int]:
    mapping: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = str(row["_y"].get(target) or "")
        if label:
            mapping[label].add(str(row.get("experiment_id") or ""))
    return {label: len(groups) for label, groups in mapping.items()}


def evaluate_window_fusion_stage(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    feature_mode: str,
    output_dir: str | Path,
    parent_description: str,
    window_size_sec: float,
    fisher_top_k: int = FISHER_TOP_K,
) -> Dict[str, Any]:
    """Grouped OOF window classifier followed by client-trace late fusion.

    Fisher selection is performed inside each training fold. All windows from
    the same experiment remain in the same fold. The fused final prediction is
    therefore based exclusively on OOF probabilities for that held-out run.
    """
    ml = _require_ml()
    RandomForestClassifier = ml["RandomForestClassifier"]
    StratifiedGroupKFold = ml["StratifiedGroupKFold"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    usable = [row for row in rows if str(row["_y"].get(target) or "").strip()]
    classes = sorted({str(row["_y"][target]) for row in usable})
    groups_per_class = _groups_per_class(usable, target)
    min_groups = min(groups_per_class.values()) if groups_per_class else 0

    if len(classes) < 2 or min_groups < 2:
        result = {
            "status": "insufficient_independent_runs",
            "target": target,
            "feature_mode": feature_mode,
            "window_size_sec": window_size_sec,
            "classes": classes,
            "independent_groups_per_class": groups_per_class,
            "window_sample_count": len(usable),
            "parent": parent_description,
        }
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    fieldnames = [key for key in usable[0] if key != "_y"]
    candidates = candidate_feature_columns(fieldnames, feature_mode)
    labels = np.asarray([str(row["_y"][target]) for row in usable], dtype=object)
    groups = np.asarray([str(row.get("experiment_id") or "") for row in usable], dtype=object)
    n_splits = min(5, min_groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    class_index = {label: i for i, label in enumerate(classes)}

    oof_pred = np.empty(len(usable), dtype=object)
    oof_prob = np.zeros((len(usable), len(classes)), dtype=float)
    seen = np.zeros(len(usable), dtype=bool)
    selected_by_fold = []
    dummy = np.zeros((len(usable), 1), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy, labels, groups), start=1):
        train_rows = [usable[i] for i in train_idx]
        X_train_all = _matrix(train_rows, candidates)
        y_train = labels[train_idx]
        ranking = fisher_score_ranking(X_train_all, y_train, candidates)
        selected = [item["feature"] for item in ranking[: max(1, min(fisher_top_k, len(ranking)))]]
        selected_by_fold.append({"fold": fold, "selected_features": selected, "ranking": ranking})

        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(_matrix(train_rows, selected), y_train)
        X_test = _matrix([usable[i] for i in test_idx], selected)
        pred = model.predict(X_test)
        raw_prob = model.predict_proba(X_test)
        aligned = np.full((len(test_idx), len(classes)), 1e-15, dtype=float)
        for source_index, label in enumerate(model.classes_):
            aligned[:, class_index[str(label)]] = raw_prob[:, source_index]
        aligned /= aligned.sum(axis=1, keepdims=True)
        oof_pred[test_idx] = pred
        oof_prob[test_idx] = aligned
        seen[test_idx] = True

    if not bool(np.all(seen)):
        raise RoundFingerprintingError("Grouped OOF did not cover every realtime window")

    truth = [str(v) for v in labels]
    pred = [str(v) for v in oof_pred]
    window_metrics = _metrics_from_predictions(truth, pred, oof_prob, classes)
    window_metrics.update({
        "status": "evaluated",
        "evaluation_unit": f"{window_size_sec:g}s_window",
        "grouping": "experiment_id",
        "fisher_selection": "inside_each_training_fold",
        "target": target,
        "parent": parent_description,
        "feature_mode": feature_mode,
        "window_size_sec": window_size_sec,
        "folds": n_splits,
        "independent_groups_per_class": groups_per_class,
    })
    (output_dir / "window_oof_metrics.json").write_text(json.dumps(window_metrics, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "selected_features_by_fold.json").write_text(json.dumps(selected_by_fold, indent=2, sort_keys=True), encoding="utf-8")

    # Preserve window-level OOF predictions because they are the evidence used
    # by the final fusion decision.
    with (output_dir / "window_oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "row_id", "experiment_id", "resolved_client_id", "window_index",
            "true_label", "predicted_label", *[f"prob_{c}" for c in classes],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(usable):
            y = row["_y"]
            writer.writerow({
                "row_id": row.get("row_id"),
                "experiment_id": row.get("experiment_id"),
                "resolved_client_id": y.get("resolved_client_id") or y.get("client_capture_id") or row.get("client_capture_id"),
                "window_index": row.get("window_index"),
                "true_label": truth[i],
                "predicted_label": pred[i],
                **{f"prob_{c}": float(oof_prob[i, class_index[c]]) for c in classes},
            })

    # Late fusion across all windows belonging to one held-out client trace.
    grouped: Dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(usable):
        y = row["_y"]
        client_id = str(y.get("resolved_client_id") or y.get("client_capture_id") or row.get("client_capture_id") or "")
        grouped[(str(row.get("experiment_id") or ""), client_id)].append(i)

    fusion_truth = []
    fusion_pred = []
    fusion_prob = []
    fusion_rows = []
    for (experiment_id, client_id), indices in sorted(grouped.items()):
        labels_for_trace = {truth[i] for i in indices}
        if len(labels_for_trace) != 1:
            raise RoundFingerprintingError(
                f"Conflicting labels across windows for {experiment_id}/{client_id}: {sorted(labels_for_trace)}"
            )
        averaged = np.mean(oof_prob[indices], axis=0)
        predicted = classes[int(np.argmax(averaged))]
        true_label = next(iter(labels_for_trace))
        fusion_truth.append(true_label)
        fusion_pred.append(predicted)
        fusion_prob.append(averaged)
        fusion_rows.append({
            "experiment_id": experiment_id,
            "resolved_client_id": client_id,
            "window_size_sec": window_size_sec,
            "window_count": len(indices),
            "true_label": true_label,
            "predicted_label": predicted,
            **{f"prob_{c}": float(averaged[class_index[c]]) for c in classes},
        })

    fusion_prob = np.asarray(fusion_prob, dtype=float)
    metrics = _metrics_from_predictions(fusion_truth, fusion_pred, fusion_prob, classes)
    metrics.update({
        "status": "evaluated",
        "target": target,
        "parent": parent_description,
        "feature_mode": feature_mode,
        "window_size_sec": window_size_sec,
        "evaluation_unit": "complete_client_trace_after_window_probability_fusion",
        "fusion_method": "mean_oof_probability_across_windows",
        "window_predictions_are_oof": True,
        "grouping": "experiment_id",
        "window_oof_metrics": window_metrics,
    })
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "fused_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(fusion_rows[0].keys()) if fusion_rows else ["experiment_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(fusion_rows)

    _write_confusion_artifacts(output_dir, fusion_truth, fusion_pred, classes, unit_name="client traces")
    _write_roc_pr(output_dir, fusion_truth, fusion_prob, classes)
    return metrics


def evaluate_temporal_fusion_hierarchy(
    *,
    x_csv: str | Path,
    y_csv: str | Path,
    output_root: str | Path,
    window_sizes_sec: Sequence[float],
    feature_modes: Sequence[str] = ("full", "size_normalized"),
) -> Dict[str, Any]:
    output_root = Path(output_root)
    summary = {}
    for feature_mode in feature_modes:
        by_scale = {}
        for scale in window_sizes_sec:
            rows = _load_window_rows(x_csv, y_csv, window_size_sec=float(scale))
            scale_token = ("%g" % float(scale)).replace(".", "p") + "s"
            scale_root = output_root / feature_mode / scale_token
            results = {}
            results["family"] = evaluate_window_fusion_stage(
                rows,
                target="family",
                feature_mode=feature_mode,
                output_dir=scale_root / "family" / "all",
                parent_description="all",
                window_size_sec=float(scale),
            )
            families = sorted({str(r["_y"].get("family") or "") for r in rows if r["_y"].get("family")})
            arch = {}
            for family in families:
                subset = [r for r in rows if str(r["_y"].get("family") or "") == family]
                arch[family] = evaluate_window_fusion_stage(
                    subset,
                    target="architecture",
                    feature_mode=feature_mode,
                    output_dir=scale_root / "architecture" / _safe_slug(family),
                    parent_description=f"family={family}; conditional_on_true_parent",
                    window_size_sec=float(scale),
                )
            results["architecture_by_family"] = arch
            by_scale[str(scale)] = results
        summary[feature_mode] = by_scale
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "temporal_fusion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary_json": str(summary_path)}
