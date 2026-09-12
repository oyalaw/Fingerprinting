#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

DEFAULT_ENDPOINT_PATTERNS = (
    "*resource*.csv",
    "*round_metrics*.csv",
    "*client_update_metrics*.csv",
    "*anomaly_metrics*.csv",
    "*anomaly_detection_metrics*.csv",
)

BLOCKED_NAME_TOKENS = {
    "family", "architecture", "variant", "framework", "runtime", "dataset",
    "application", "model_name", "model_family", "model_architecture",
    "client_id", "client", "server_id", "role", "experiment_id", "run_id",
    "trace_id", "trace", "src_ip", "dst_ip", "ip", "src_port", "dst_port",
    "port", "hostname", "host", "device", "partition", "partition_id",
    "round", "round_id", "round_index", "epoch", "timestamp", "time_utc",
    "phase", "label", "target", "class",
}

SAFE_ENDPOINT_HINTS = (
    "cpu", "memory", "ram", "gpu", "util", "power",
    "bytes_sent", "bytes_received", "network", "throughput",
    "download_time", "upload_time", "training_time", "transaction_time",
    "aggregation_time", "model_size", "update_size", "norm",
    "loss", "accuracy", "precision", "recall", "f1", "auroc", "auprc",
    "clients_received", "clients_selected",
)

LABEL_COLS = (
    "family", "architecture", "variant",
    "application", "dataset", "framework", "runtime",
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="collected_experiments")
    p.add_argument("--x", default="fingerprinting_dataset/fingerprinting_X_proxy.csv")
    p.add_argument("--y", default="fingerprinting_dataset/fingerprinting_Y_ground_truth.csv")
    p.add_argument("--out-dir", default="fingerprinting_dataset/endpoint_assisted")
    return p.parse_args()

def canonical_experiment_id(path: Path):
    for part in reversed(path.parts):
        if part.startswith("run_"):
            return part
    return None

def normalize_client_id(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    m = re.search(r"client[_\-\s]?(\d+)", s)
    if m:
        return f"client_{int(m.group(1))}"
    if s.isdigit():
        return f"client_{int(s)}"
    return s if s.startswith("client_") else None

def is_numeric_series(s: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(s)
        or pd.to_numeric(s, errors="coerce").notna().mean() >= 0.90
    )

def endpoint_predictor_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        lc = str(c).strip().lower()
        tokens = set(re.split(r"[^a-z0-9]+", lc))
        if any(tok in BLOCKED_NAME_TOKENS for tok in tokens if tok):
            if not any(h in lc for h in SAFE_ENDPOINT_HINTS):
                continue
        if not any(h in lc for h in SAFE_ENDPOINT_HINTS):
            continue
        if is_numeric_series(df[c]):
            out.append(c)
    return out

def aggregate_numeric(df: pd.DataFrame, cols: List[str], prefix: str) -> Dict[str, float]:
    feats = {}
    for c in cols:
        vals = (
            pd.to_numeric(df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if vals.empty:
            continue
        key = re.sub(r"[^a-zA-Z0-9_]+", "_", str(c)).strip("_").lower()
        feats[f"{prefix}{key}__mean"] = float(vals.mean())
        feats[f"{prefix}{key}__std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        feats[f"{prefix}{key}__median"] = float(vals.median())
        feats[f"{prefix}{key}__min"] = float(vals.min())
        feats[f"{prefix}{key}__max"] = float(vals.max())
        if len(vals) > 1:
            feats[f"{prefix}{key}__delta"] = float(vals.iloc[-1] - vals.iloc[0])
    return feats

def discover_endpoint_tables(root: Path):
    by_run_client = defaultdict(list)
    by_run_server = defaultdict(list)
    seen = set()

    for pat in DEFAULT_ENDPOINT_PATTERNS:
        for path in root.rglob(pat):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            run_id = canonical_experiment_id(path)
            if not run_id:
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue

            role = None
            if "role" in df.columns:
                vals = {str(x).strip().lower() for x in df["role"].dropna().unique()}
                if "server" in vals:
                    role = "server"
                elif "client" in vals:
                    role = "client"

            if role is None:
                pl = str(path).lower()
                role = "server" if "/server/" in pl or "_server_" in pl else "client"

            if role == "server":
                by_run_server[run_id].append((path, df))
                continue

            client_col = next(
                (c for c in df.columns if str(c).lower() in {"client_id", "client"}),
                None,
            )
            if client_col is not None:
                for raw_id, sub in df.groupby(client_col, dropna=True):
                    cid = normalize_client_id(raw_id)
                    if cid:
                        by_run_client[(run_id, cid)].append((path, sub.copy()))
            else:
                cid = normalize_client_id(str(path))
                if cid:
                    by_run_client[(run_id, cid)].append((path, df))

    return by_run_client, by_run_server

def load_y_map(y_path: Path):
    y = pd.read_csv(y_path)
    run_col = next(
        (
            c for c in (
                "experiment_id",
                "run_id",
            )
            if c in y.columns
        ),
        None,
    )

    client_col = next(
        (
            c for c in (
                "client_capture_id",
                "resolved_client_id",
                "client_id",
                "client",
            )
            if c in y.columns
        ),
        None,
    )
    if run_col is None or client_col is None:
        raise RuntimeError("Y file must contain experiment/run and client identifiers.")

    keep = [run_col, client_col] + [c for c in LABEL_COLS if c in y.columns]
    y = y[keep].drop_duplicates()

    out = {}
    for _, r in y.iterrows():
        run = str(r[run_col])
        cid = normalize_client_id(r[client_col])
        if cid:
            out[(run, cid)] = {c: r[c] for c in LABEL_COLS if c in y.columns}
    return out

def iter_proxy_samples(x_path: Path, y_map):
    chosen = {}
    for chunk in pd.read_csv(x_path, chunksize=100_000):
        run_col = next(
            (
                c for c in (
                    "experiment_id",
                    "run_id",
                )
                if c in chunk.columns
            ),
            None,
        )

        client_col = next(
            (
                c for c in (
                    "client_capture_id",
                    "resolved_client_id",
                    "client_id",
                    "client",
                )
                if c in chunk.columns
            ),
            None,
        )
        if run_col is None or client_col is None:
            raise RuntimeError("X file must contain experiment/run and client identifiers.")

        sub = chunk

        if "row_type" in chunk.columns:
            sub = chunk.loc[
                chunk["row_type"]
                .astype(str)
                .str.lower()
                .eq("overall")
            ]

        elif "window_type" in chunk.columns:
            sub = chunk.loc[
                chunk["window_type"]
                .astype(str)
                .str.lower()
                .eq("overall")
            ]

        for _, row in sub.iterrows():
            run = str(row[run_col])
            cid = normalize_client_id(row[client_col])
            key = (run, cid)
            if cid and key in y_map and key not in chosen:
                chosen[key] = row.to_dict()
    return chosen

def proxy_predictor_columns(row: Dict) -> List[str]:
    """Return only canonical proxy-observable fingerprint predictors.

    In fingerprinting_X_proxy.csv the feature block begins at
    packet_count_total. Columns before that point are identifiers,
    grouping metadata, window coordinates, or quality metadata and
    must not enter the student classifier.
    """
    columns = list(row.keys())

    if "packet_count_total" not in columns:
        raise RuntimeError(
            "Canonical proxy feature start "
            "'packet_count_total' was not found."
        )

    start_index = columns.index(
        "packet_count_total"
    )

    candidates = columns[
        start_index:
    ]

    blocked = {
        "experiment_id",
        "run_id",
        "client_capture_id",
        "resolved_client_id",
        "client_id",
        "client",
        "row_id",
        "row_type",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "window_size_sec",
        "trace_start_offset_sec",
        "trace_end_offset_sec",
        "window_start_global_sec",
        "window_end_global_sec",
        "packet_information_threshold",
        "packet_information_ok",
        *LABEL_COLS,
    }

    out = []

    for c in candidates:
        lc = str(c).lower()

        if lc in blocked:
            continue

        if any(
            token in lc
            for token in (
                "ground_truth",
                "partition",
                "hostname",
                "device",
            )
        ):
            continue

        value = row.get(c)

        try:
            fv = float(value)
        except Exception:
            continue

        if math.isfinite(fv):
            out.append(c)

    return out

def main():
    args = parse_args()
    root = Path(args.root).resolve()
    x_path = Path(args.x)
    y_path = Path(args.y)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_map = load_y_map(y_path)
    client_tables, server_tables = discover_endpoint_tables(root)
    proxy_rows = iter_proxy_samples(x_path, y_map)

    if not proxy_rows:
        raise RuntimeError("No proxy samples were found.")

    proxy_cols = proxy_predictor_columns(next(iter(proxy_rows.values())))

    records = []
    missing_endpoint = []

    for key in sorted(proxy_rows):
        run_id, client_id = key
        labels = y_map[key]
        teacher = {}

        for _, df in client_tables.get(key, []):
            cols = endpoint_predictor_columns(df)
            teacher.update(aggregate_numeric(df, cols, "teacher_client_"))

        for _, df in server_tables.get(run_id, []):
            cols = endpoint_predictor_columns(df)
            teacher.update(aggregate_numeric(df, cols, "teacher_server_"))

        if not teacher:
            missing_endpoint.append({
                "experiment_id": run_id,
                "client_id": client_id,
            })
            continue

        rec = {
            "experiment_id": run_id,
            "client_id": client_id,
            **labels,
            **teacher,
        }

        prow = proxy_rows[key]
        for c in proxy_cols:
            try:
                rec[f"proxy_{c}"] = float(prow[c])
            except Exception:
                rec[f"proxy_{c}"] = np.nan

        records.append(rec)

    if not records:
        raise RuntimeError("No paired endpoint/proxy samples could be constructed.")

    df = pd.DataFrame(records)

    teacher_cols = [
        c for c in df.columns
        if c.startswith("teacher_") and df[c].notna().any()
    ]
    proxy_cols_out = [
        c for c in df.columns
        if c.startswith("proxy_") and df[c].notna().any()
    ]

    final_cols = (
        ["experiment_id", "client_id"]
        + [c for c in LABEL_COLS if c in df.columns]
        + teacher_cols
        + proxy_cols_out
    )
    df = df[final_cols]

    out_csv = out_dir / "endpoint_assisted_paired.csv"
    df.to_csv(out_csv, index=False)

    family_counts = {}
    if "family" in df.columns:
        family_counts = (
            df[["experiment_id", "family"]]
            .drop_duplicates()
            .groupby("family")["experiment_id"]
            .nunique()
            .to_dict()
        )

    audit = {
        "paired_samples": int(len(df)),
        "paired_experiments": int(df["experiment_id"].nunique()),
        "paired_client_traces": int(
            df[["experiment_id", "client_id"]].drop_duplicates().shape[0]
        ),
        "teacher_predictor_count": int(len(teacher_cols)),
        "proxy_predictor_count": int(len(proxy_cols_out)),
        "family_independent_experiments": {
            str(k): int(v) for k, v in family_counts.items()
        },
        "missing_endpoint_count": int(len(missing_endpoint)),
        "missing_endpoint": missing_endpoint,
        "output_csv": str(out_csv),
        "guardrails": {
            "grouping_unit": "experiment_id",
            "teacher_features": "endpoint-only privileged training information",
            "student_features": "proxy-only",
            "endpoint_features_used_at_student_test_time": False,
        },
    }

    (out_dir / "endpoint_assisted_dataset_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    schema = {
        "teacher_columns": teacher_cols,
        "proxy_columns": proxy_cols_out,
        "label_columns": [c for c in LABEL_COLS if c in df.columns],
    }
    (out_dir / "endpoint_assisted_schema.json").write_text(
        json.dumps(schema, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
