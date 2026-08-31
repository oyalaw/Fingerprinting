from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_fingerprint.architecture_models import (
    ArchitectureModelError,
    model_directory_name,
    train_hierarchy_bundle,
)


def _find_dataset(root: Path):
    candidates = [
        (
            root / "fingerprinting_dataset/fingerprinting_X_proxy.csv",
            root / "fingerprinting_dataset/fingerprinting_Y_ground_truth.csv",
        )
    ]
    for x_path in root.rglob("*_X_proxy.csv"):
        y_path = Path(
            str(x_path).replace("_X_proxy.csv", "_Y_ground_truth.csv")
        )
        candidates.append((x_path, y_path))

    seen = set()
    for x_path, y_path in candidates:
        key = (x_path.resolve(), y_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if x_path.exists() and y_path.exists():
            return x_path, y_path
    return None, None


def _window_sizes(x_path: Path):
    values = set()
    with x_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("row_type") != "window":
                continue
            value = row.get("window_size_sec")
            try:
                number = float(value) if value not in {None, ""} else 0.0
            except ValueError:
                number = 0.0
            if number <= 0:
                try:
                    number = (
                        float(row.get("window_end_sec", 0))
                        - float(row.get("window_start_sec", 0))
                    )
                except ValueError:
                    number = 0.0
            if number > 0:
                values.add(number)
    return sorted(values)



def _metric_rows(trained):
    rows = []
    for result in trained:
        common = {
            "feature_mode": result.get("feature_mode"),
            "mode": result.get("mode"),
            "window_size_sec": result.get("window_size_sec"),
        }
        stages = result.get("stages", {})
        entries = [("family", "all", stages.get("family", {}))]
        entries.extend(
            ("architecture", parent, stage)
            for parent, stage in stages.get("architecture_by_family", {}).items()
        )
        entries.extend(
            ("variant", parent, stage)
            for parent, stage in stages.get("variant_by_parent", {}).items()
        )
        entries.extend(
            ("application", parent, stage)
            for parent, stage in stages.get("application_by_parent", {}).items()
        )
        for level, parent, stage in entries:
            evaluation = stage.get("evaluation", {}) or {}
            rows.append({
                **common,
                "level": level,
                "parent": parent,
                "kind": stage.get("kind"),
                "status": evaluation.get("status", stage.get("kind")),
                "sample_count": stage.get("sample_count"),
                "accuracy_mean": evaluation.get("accuracy_mean"),
                "accuracy_std": evaluation.get("accuracy_std"),
                "balanced_accuracy_mean": evaluation.get("balanced_accuracy_mean"),
                "balanced_accuracy_std": evaluation.get("balanced_accuracy_std"),
                "precision_mean": evaluation.get("macro_precision_mean"),
                "precision_std": evaluation.get("macro_precision_std"),
                "recall_mean": evaluation.get("macro_recall_mean"),
                "recall_std": evaluation.get("macro_recall_std"),
                "f1_mean": evaluation.get("macro_f1_mean"),
                "f1_std": evaluation.get("macro_f1_std"),
                "loss_mean": evaluation.get("log_loss_mean"),
                "loss_std": evaluation.get("log_loss_std"),
            })
    return rows


def _write_metrics_csv(model_root: Path, trained):
    rows = _metric_rows(trained)
    path = model_root / "hierarchical_metrics.csv"
    fieldnames = [
        "feature_mode", "mode", "window_size_sec", "level", "parent",
        "kind", "status", "sample_count", "accuracy_mean", "accuracy_std",
        "balanced_accuracy_mean", "balanced_accuracy_std",
        "precision_mean", "precision_std", "recall_mean", "recall_std",
        "f1_mean", "f1_std", "loss_mean", "loss_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

def main() -> None:
    root = Path(".").resolve()
    x_path, y_path = _find_dataset(root)
    if x_path is None or y_path is None:
        raise SystemExit(
            "No prepared fingerprinting X/Y dataset found. First run:\n"
            "  python3 prepare_fingerprinting_dataset.py"
        )

    model_root = root / "fingerprinting_models"
    model_root.mkdir(parents=True, exist_ok=True)

    print(f"X: {x_path}")
    print(f"Y: {y_path}")
    print(f"Models: {model_root}")

    window_sizes = _window_sizes(x_path)
    print(
        "Real-time window scales: "
        + (
            ", ".join(f"{value:g}s" for value in window_sizes)
            if window_sizes
            else "none"
        )
    )

    trained = []
    failures = []

    for feature_mode in ("full", "size_normalized"):
        jobs = [("final", None)]
        jobs.extend(("realtime", value) for value in window_sizes)

        for mode, window_size in jobs:
            output_dir = (
                model_root
                / model_directory_name(
                    mode,
                    feature_mode,
                    window_size,
                )
            )
            try:
                result = train_hierarchy_bundle(
                    x_csv=x_path,
                    y_csv=y_path,
                    output_dir=output_dir,
                    mode=mode,
                    feature_mode=feature_mode,
                    window_size_sec=window_size,
                )
                trained.append(result)
                label = (
                    "final"
                    if mode == "final"
                    else f"realtime {window_size:g}s"
                )
                print(
                    f"[trained] {feature_mode:15} {label:16} "
                    f"samples={result['sample_count']} "
                    f"experiments={result['experiment_count']}"
                )
            except ArchitectureModelError as exc:
                failures.append(
                    {
                        "feature_mode": feature_mode,
                        "mode": mode,
                        "window_size_sec": window_size,
                        "error": str(exc),
                    }
                )
                print(
                    f"[skipped] {feature_mode} {mode} "
                    f"{window_size}: {exc}"
                )

    metrics_path = _write_metrics_csv(model_root, trained)
    print(f"Hierarchical metrics: {metrics_path}")

    summary = {
        "x_proxy_csv": str(x_path),
        "y_ground_truth_csv": str(y_path),
        "model_root": str(model_root),
        "hierarchical_metrics_csv": str(metrics_path),
        "fisher_top_k": 10,
        "window_sizes_sec": window_sizes,
        "trained": trained,
        "failures": failures,
        "important_evaluation_rule": (
            "Publication metrics must use independent experiment/run groups. "
            "Windows from the same run must never be randomly split between "
            "training and test."
        ),
    }
    summary_path = model_root / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nTraining summary: {summary_path}")


if __name__ == "__main__":
    main()
