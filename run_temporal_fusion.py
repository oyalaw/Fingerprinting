from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ai_fingerprint.temporal_fusion import evaluate_temporal_fusion_hierarchy


def discover_scales(x_csv: Path):
    values = set()
    with x_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("row_type") or "") != "window":
                continue
            try:
                value = float(row.get("window_size_sec") or 0.0)
            except ValueError:
                value = 0.0
            if value > 0:
                values.add(value)
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Late-fuse grouped OOF realtime-window probabilities into one "
            "final client-trace fingerprint prediction. Existing models are untouched."
        )
    )
    parser.add_argument("--x", default="fingerprinting_dataset/fingerprinting_X_proxy.csv")
    parser.add_argument("--y", default="fingerprinting_dataset/fingerprinting_Y_ground_truth.csv")
    parser.add_argument("--output", default="fingerprinting_results/temporal_fusion")
    parser.add_argument(
        "--scales",
        default="",
        help="Comma-separated seconds. Empty means use every scale present in X.",
    )
    args = parser.parse_args()

    x_path = Path(args.x)
    y_path = Path(args.y)
    if not x_path.exists() or not y_path.exists():
        raise SystemExit("Prepared fingerprinting X/Y dataset not found.")

    if args.scales.strip():
        scales = [float(v.strip()) for v in args.scales.split(",") if v.strip()]
    else:
        scales = discover_scales(x_path)
    if not scales:
        raise SystemExit("No realtime window scales found.")

    print("Temporal fusion scales: " + ", ".join(f"{v:g}s" for v in scales))
    result = evaluate_temporal_fusion_hierarchy(
        x_csv=x_path,
        y_csv=y_path,
        output_root=Path(args.output),
        window_sizes_sec=scales,
    )
    print(f"Summary: {result['summary_json']}")
    print("Existing complete-trace and realtime classifiers were not modified.")


if __name__ == "__main__":
    main()
