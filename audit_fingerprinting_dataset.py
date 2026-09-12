from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd().resolve()
X_PATH = ROOT / "fingerprinting_dataset" / "fingerprinting_X_proxy.csv"
Y_PATH = ROOT / "fingerprinting_dataset" / "fingerprinting_Y_ground_truth.csv"
OUT_DIR = ROOT / "fingerprinting_dataset" / "audit"


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def client_id(row):
    return str(row.get("resolved_client_id") or row.get("client_capture_id") or "").strip()


def representation(row):
    rt = str(row.get("row_type", "")).strip().lower()
    if rt == "overall":
        return "aggregated"
    ws = as_float(row.get("window_size_sec"), -1.0)
    return f"{ws:g}s" if ws > 0 else "unknown"


def main():
    if not X_PATH.exists() or not Y_PATH.exists():
        raise SystemExit(
            "Dataset files not found. Expected:\n"
            f"  {X_PATH}\n  {Y_PATH}"
        )

    with Y_PATH.open(newline="", encoding="utf-8") as f:
        y_rows = list(csv.DictReader(f))
    y_by_id = {str(r["row_id"]): r for r in y_rows}

    with X_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        x_fields = reader.fieldnames or []
        x_rows = list(reader)

    joined = []
    missing_y = 0
    for x in x_rows:
        y = y_by_id.get(str(x.get("row_id", "")))
        if y is None:
            missing_y += 1
            continue
        row = dict(x)
        row.update({f"Y::{k}": v for k, v in y.items()})
        joined.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reps = defaultdict(list)
    for row in joined:
        reps[representation(row)].append(row)

    summary_rows = []
    for rep in ["0.5s", "1s", "2s", "5s", "aggregated", "unknown"]:
        rows = reps.get(rep, [])
        if not rows:
            continue
        experiments = {str(r.get("experiment_id", "")).strip() for r in rows}
        traces = {
            (str(r.get("experiment_id", "")).strip(), client_id(r))
            for r in rows
        }
        sparse = sum(
            1 for r in rows
            if str(r.get("packet_information_ok", "")).strip() in {"0", "0.0", "False", "false"}
        )
        summary_rows.append({
            "representation": rep,
            "classification_rows": len(rows),
            "independent_experiments": len(experiments - {""}),
            "client_traces": len({t for t in traces if t[0] and t[1]}),
            "sparse_rows": sparse,
            "sparse_fraction": sparse / len(rows) if rows else 0.0,
        })

    with (OUT_DIR / "representation_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_rows = []
    grouped = defaultdict(list)
    for r in joined:
        key = (
            representation(r),
            str(r.get("Y::family", "")),
            str(r.get("Y::architecture", "")),
            str(r.get("Y::variant", "")),
        )
        grouped[key].append(r)

    for (rep, fam, arch, var), rows in sorted(grouped.items()):
        exps = {str(r.get("experiment_id", "")) for r in rows if r.get("experiment_id")}
        traces = {(str(r.get("experiment_id", "")), client_id(r)) for r in rows}
        detail_rows.append({
            "representation": rep,
            "family": fam,
            "architecture": arch,
            "variant": var,
            "rows": len(rows),
            "experiments": len(exps),
            "client_traces": len({t for t in traces if t[0] and t[1]}),
        })

    if detail_rows:
        with (OUT_DIR / "representation_inventory_detail.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    # Window-rate sanity audit. For every non-empty denominator, the derived
    # rates must match counts/bytes divided by the actual interval represented
    # by window_end_sec - window_start_sec.
    checks = [
        ("packets_per_second", "packet_count_total"),
        ("bytes_per_second", "bytes_total"),
        ("upload_packets_per_second", "packet_count_up"),
        ("download_packets_per_second", "packet_count_down"),
        ("upload_bytes_per_second", "bytes_up"),
        ("download_bytes_per_second", "bytes_down"),
    ]
    mismatches = []
    checked = 0
    for r in joined:
        if str(r.get("row_type", "")).lower() != "window":
            continue
        start = as_float(r.get("window_start_sec"))
        end = as_float(r.get("window_end_sec"))
        interval = end - start
        if interval <= 0:
            continue
        checked += 1
        for rate_name, total_name in checks:
            if rate_name not in r or total_name not in r:
                continue
            actual = as_float(r.get(rate_name))
            expected = as_float(r.get(total_name)) / interval
            tol = max(1e-8, abs(expected) * 1e-8)
            if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=tol):
                if len(mismatches) < 100:
                    mismatches.append({
                        "row_id": r.get("row_id"),
                        "experiment_id": r.get("experiment_id"),
                        "client": client_id(r),
                        "representation": representation(r),
                        "feature": rate_name,
                        "total_feature": total_name,
                        "interval_sec": interval,
                        "actual": actual,
                        "expected": expected,
                    })

    if mismatches:
        with (OUT_DIR / "window_rate_mismatches.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(mismatches[0].keys()))
            writer.writeheader()
            writer.writerows(mismatches)

    forbidden_label_names = {
        "family", "architecture", "variant", "application", "dataset",
        "task", "framework", "runtime", "device", "operating_system",
        "loss", "accuracy", "precision", "recall", "f1", "f1_score",
    }
    leaked = sorted(forbidden_label_names.intersection(x_fields))

    report = {
        "x_rows": len(x_rows),
        "y_rows": len(y_rows),
        "joined_rows": len(joined),
        "missing_y_rows": missing_y,
        "representations": summary_rows,
        "window_rows_checked": checked,
        "window_rate_mismatch_examples": len(mismatches),
        "window_rate_sanity_pass": not mismatches,
        "forbidden_label_columns_present_in_x": leaked,
        "label_leakage_pass": not leaked,
    }
    (OUT_DIR / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Dataset audit")
    print("=" * 90)
    for item in summary_rows:
        print(
            f"{item['representation']:>10}: rows={item['classification_rows']:,} "
            f"experiments={item['independent_experiments']} "
            f"traces={item['client_traces']} "
            f"sparse={item['sparse_fraction']:.2%}"
        )
    print("\nWindow-rate sanity:", "PASS" if not mismatches else "FAIL")
    print("Forbidden label columns in X:", leaked or "none")
    print("Audit JSON:", OUT_DIR / "dataset_audit.json")
    print("Inventory CSV:", OUT_DIR / "representation_inventory.csv")
    if mismatches:
        print("Mismatch examples:", OUT_DIR / "window_rate_mismatches.csv")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
