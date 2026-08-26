from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_fingerprint.architecture_models import (
    ArchitectureModelError,
    DEFAULT_FISHER_MIN_FEATURES,
    DEFAULT_FISHER_MIN_SCORE,
    DEFAULT_FISHER_TOP_K,
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


def _print_stage_selection(result):
    stages = result["stages"]

    def show(label, stage):
        if stage.get("kind") != "classifier":
            print(
                f"    {label}: {stage.get('kind')} "
                f"classes={stage.get('classes', [])}"
            )
            return
        selected = stage.get("feature_columns", [])
        ranking = stage.get("fisher_ranking", [])
        top = ", ".join(
            f"{item['feature']}={float(item['fisher_score']):.3g}"
            for item in ranking[:5]
        )
        print(
            f"    {label}: selected={len(selected)} "
            f"top Fisher [{top}]"
        )

    show("family", stages["family"])
    for parent, stage in stages["architecture_by_family"].items():
        show(f"architecture|{parent}", stage)
    for parent, stage in stages["variant_by_parent"].items():
        show(f"variant|{parent}", stage)


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
    print(
        "Feature selection: class-balanced Fisher, "
        f"top_k={DEFAULT_FISHER_TOP_K}, "
        f"min_score={DEFAULT_FISHER_MIN_SCORE:g}, "
        f"min_features={DEFAULT_FISHER_MIN_FEATURES}"
    )

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
                _print_stage_selection(result)
                print(
                    f"    Fisher CSV: "
                    f"{result['fisher_scores_csv']}"
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

    summary = {
        "x_proxy_csv": str(x_path),
        "y_ground_truth_csv": str(y_path),
        "model_root": str(model_root),
        "window_sizes_sec": window_sizes,
        "feature_selection": {
            "method": "class_balanced_fisher",
            "stage_specific": True,
            "top_k": DEFAULT_FISHER_TOP_K,
            "min_score": DEFAULT_FISHER_MIN_SCORE,
            "min_features": DEFAULT_FISHER_MIN_FEATURES,
            "grouped_evaluation_rule": (
                "Fisher selection is re-fitted inside each training fold."
            ),
        },
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
