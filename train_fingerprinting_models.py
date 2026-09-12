from __future__ import annotations

"""Single-command trainer for 0.5 s, 1 s, 2 s, 5 s, and aggregated fingerprints.

This script is intentionally argument-free for the normal central workflow.
`run_central_fingerprinting.py` can continue to call it exactly as before.

Outputs:
  fingerprinting_results/multi_representation/
      evaluation_summary.csv
      representation_inventory.csv
      dataset_inventory.csv
      representation_comparison.csv
      comparison_plots/*.pdf|png
      size_normalized/<representation>/<hierarchy>/<parent>/...
"""

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
X_PATH = PROJECT_ROOT / "fingerprinting_dataset" / "fingerprinting_X_proxy.csv"
Y_PATH = PROJECT_ROOT / "fingerprinting_dataset" / "fingerprinting_Y_ground_truth.csv"
OUTPUT_ROOT = PROJECT_ROOT / "fingerprinting_results" / "multi_representation"
EVALUATOR = PROJECT_ROOT / "generate_oof_fingerprinting_results.py"

REP_ORDER = ["0.5 s", "1 s", "2 s", "5 s", "Aggregated"]
REP_INDEX = {name: i for i, name in enumerate(REP_ORDER)}


def normalize_representation(row_type: str, window: str) -> str | None:
    row_type = str(row_type or "").strip().lower()
    if row_type == "overall":
        return "Aggregated"
    if row_type != "window":
        return None
    try:
        value = float(window)
    except (TypeError, ValueError):
        return None
    for target, label in [(0.5, "0.5 s"), (1.0, "1 s"), (2.0, "2 s"), (5.0, "5 s")]:
        if abs(value - target) < 1e-9:
            return label
    return None


def stream_dataset_inventory() -> tuple[Path, Path, dict[str, Any]]:
    if not X_PATH.exists() or not Y_PATH.exists():
        raise SystemExit(
            "Fingerprinting X/Y files are missing. Run run_central_fingerprinting.py first."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    by_rep = Counter()
    experiments_by_rep: dict[str, set[str]] = defaultdict(set)
    traces_by_rep: dict[str, set[tuple[str, str]]] = defaultdict(set)
    by_rep_family = Counter()
    by_rep_architecture = Counter()
    by_rep_variant = Counter()
    by_rep_client = Counter()

    with X_PATH.open(newline="", encoding="utf-8") as xf, Y_PATH.open(
        newline="", encoding="utf-8"
    ) as yf:
        xr = csv.DictReader(xf)
        yr = csv.DictReader(yf)
        for x, y in zip(xr, yr):
            if str(x.get("row_id")) != str(y.get("row_id")):
                raise RuntimeError(
                    f"X/Y row_id mismatch: {x.get('row_id')} != {y.get('row_id')}"
                )
            rep = normalize_representation(
                x.get("row_type", ""), x.get("window_size_sec", "")
            )
            if rep is None:
                continue
            experiment = str(y.get("experiment_id", "")).strip()
            client = str(
                y.get("resolved_client_id")
                or y.get("client_capture_id")
                or "unknown"
            ).strip()
            family = str(y.get("family", "unknown")).strip()
            architecture = str(y.get("architecture", "unknown")).strip()
            variant = str(y.get("variant", "unknown")).strip()

            by_rep[rep] += 1
            experiments_by_rep[rep].add(experiment)
            traces_by_rep[rep].add((experiment, client))
            by_rep_family[(rep, family)] += 1
            by_rep_architecture[(rep, architecture)] += 1
            by_rep_variant[(rep, variant)] += 1
            by_rep_client[(rep, client)] += 1

    inventory = OUTPUT_ROOT / "dataset_inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "representation",
                "observation_count",
                "independent_experiments",
                "client_traces",
            ],
        )
        writer.writeheader()
        for rep in REP_ORDER:
            writer.writerow(
                {
                    "representation": rep,
                    "observation_count": by_rep[rep],
                    "independent_experiments": len(experiments_by_rep[rep]),
                    "client_traces": len(traces_by_rep[rep]),
                }
            )

    detail = OUTPUT_ROOT / "dataset_inventory_detail.csv"
    with detail.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["representation", "dimension", "value", "observation_count"],
        )
        writer.writeheader()
        for counter, dimension in [
            (by_rep_client, "client"),
            (by_rep_family, "family"),
            (by_rep_architecture, "architecture"),
            (by_rep_variant, "variant"),
        ]:
            for (rep, value), count in sorted(
                counter.items(), key=lambda item: (REP_INDEX.get(item[0][0], 999), item[0][1])
            ):
                writer.writerow(
                    {
                        "representation": rep,
                        "dimension": dimension,
                        "value": value,
                        "observation_count": count,
                    }
                )

    summary = {
        rep: {
            "observation_count": by_rep[rep],
            "experiment_count": len(experiments_by_rep[rep]),
            "client_trace_count": len(traces_by_rep[rep]),
        }
        for rep in REP_ORDER
    }
    return inventory, detail, summary


def run_evaluator() -> None:
    if not EVALUATOR.exists():
        raise SystemExit(f"Missing evaluator: {EVALUATOR}")
    command = [
        sys.executable,
        str(EVALUATOR),
        "--x",
        str(X_PATH),
        "--y",
        str(Y_PATH),
        "--output",
        str(OUTPUT_ROOT),
        "--mode",
        "all",
        "--feature-mode",
        "size_normalized",
        "--windows",
        "0.5,1,2,5",
        "--descriptive-fallback",
        "variants",
    ]
    print("\nRunning five-representation group-disjoint evaluation...")
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def build_comparison() -> tuple[Path, Path]:
    summary_path = OUTPUT_ROOT / "evaluation_summary.csv"
    if not summary_path.exists():
        raise RuntimeError(f"Missing evaluation summary: {summary_path}")

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    evaluated = [
        row
        for row in rows
        if row.get("status") == "evaluated"
        and row.get("feature_mode") == "size_normalized"
    ]

    comparison_path = OUTPUT_ROOT / "representation_comparison.csv"
    fields = [
        "representation",
        "level",
        "parent",
        "sample_count",
        "oof_sample_count",
        "experiment_count",
        "oof_experiment_count",
        "folds",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_auroc_ovr",
        "macro_auprc_ovr",
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(
                evaluated,
                key=lambda r: (
                    r.get("level", ""),
                    r.get("parent", ""),
                    REP_INDEX.get(r.get("representation", ""), 999),
                ),
            )
        )

    plot_dir = OUTPUT_ROOT / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in evaluated:
        grouped[(row.get("level", ""), row.get("parent", ""))].append(row)

    index_rows = []
    for (level, parent), items in sorted(grouped.items()):
        items.sort(key=lambda r: REP_INDEX.get(r.get("representation", ""), 999))
        reps = [r["representation"] for r in items]
        accuracy = [float(r["accuracy"]) for r in items]
        macro_f1 = [float(r["macro_f1"]) for r in items]

        fig, ax = plt.subplots(figsize=(8, 5.5))
        x = list(range(len(reps)))
        ax.plot(x, accuracy, marker="o", label="Accuracy")
        ax.plot(x, macro_f1, marker="o", label="Macro F1")
        ax.set_xticks(x)
        ax.set_xticklabels(reps)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.set_xlabel("Observation representation")
        ax.set_title(f"Fingerprinting performance: {level} / {parent}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        name = f"{level}__{parent}".replace("::", "__").replace("/", "_").replace(" ", "_")
        png = plot_dir / f"{name}.png"
        pdf = plot_dir / f"{name}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)

        index_rows.append(
            {
                "level": level,
                "parent": parent,
                "png": str(png),
                "pdf": str(pdf),
            }
        )

    plot_index = OUTPUT_ROOT / "comparison_plot_index.csv"
    with plot_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["level", "parent", "png", "pdf"])
        writer.writeheader()
        writer.writerows(index_rows)

    return comparison_path, plot_index


def main() -> None:
    print("Hierarchical AI fingerprinting: multi-representation evaluation")
    print("Representations: 0.5 s, 1 s, 2 s, 5 s, Aggregated")
    print("Split policy: experiment-disjoint grouped OOF")
    print("Feature selection: Fisher ranking inside training fold only")
    print("Predictors: proxy-observable network features only")

    inventory, detail, inventory_summary = stream_dataset_inventory()
    print("\nDataset inventory:")
    for rep in REP_ORDER:
        item = inventory_summary[rep]
        print(
            f"  {rep:10s}: observations={item['observation_count']:,} "
            f"experiments={item['experiment_count']} "
            f"client_traces={item['client_trace_count']}"
        )

    run_evaluator()
    comparison, plot_index = build_comparison()

    run_summary = {
        "representations": REP_ORDER,
        "dataset_inventory_csv": str(inventory),
        "dataset_inventory_detail_csv": str(detail),
        "evaluation_summary_csv": str(OUTPUT_ROOT / "evaluation_summary.csv"),
        "representation_comparison_csv": str(comparison),
        "comparison_plot_index_csv": str(plot_index),
        "policy": {
            "split": "deterministic class-balanced folds built from experiment_id groups",
            "fisher_selection": "training fold only",
            "default_feature_mode": "size_normalized",
            "unsupported_class_policy": (
                "classes with <2 independent experiments are excluded from inferential OOF "
                "metrics but retained in coverage reports; variant stages receive clearly marked "
                "DESCRIPTIVE_ONLY matrices at every representation when OOF is impossible"
            ),
        },
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )

    print("\nComplete.")
    print(f"  dataset inventory:       {inventory}")
    print(f"  evaluation summary:      {OUTPUT_ROOT / 'evaluation_summary.csv'}")
    print(f"  representation compare:  {comparison}")
    print(f"  comparison plot index:   {plot_index}")


if __name__ == "__main__":
    main()
