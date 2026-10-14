#!/usr/bin/env python3
"""
Build a label + model-metadata table for every collected FL fingerprinting run.

Usage (on the DGX):
    python3 collect_run_metadata.py /home/dgx-1/lawrence/Fingerprinting/collected_experiments \
        --out run_metadata.csv

Each run directory is expected to look like:
    <root>/<run_id>/{server,client_1,...,proxy}/...
It reads role_manifest.json, model_contract.json, config*.yaml and round_metrics.csv,
and checks whether proxy packet sequences exist. No third-party deps except PyYAML (optional).
"""
import argparse, csv, glob, json, math, os, statistics, sys

# The 19 runs in canonical_proxy_features.zip (used to report which are found / missing)
CANONICAL_RUNS = """
run_20260828t211159z_08dc7c04 run_20260828t212022z_a7ab6f4a run_20260830t165341z_94da1c16
run_20260830t171910z_fd4d81da run_20260830t183736z_8e3a8c31 run_20260830t201633z_11a6fec5
run_20260831t140006z_fb977dc7 run_20260831t141747z_8bab659b run_20260831t144320z_aead0bf7
run_20260831t154734z_6beb92ce run_20260831t165834z_d2cab726 run_20260831t171303z_5ece0500
run_20260831t173148z_3ddf9c34 run_20260831t180657z_1889c313 run_20260831t181633z_26d4f5f6
run_20260831t191517z_9182a7f9 run_20260831t225250z_fd5f756f run_20260901t192502z_7ca8363f
run_20260901t212107z_e92e8290
""".split()

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8,
               "int64": 8, "int32": 4, "int8": 1, "uint8": 1, "bool": 1}


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def load_yaml(p):
    try:
        import yaml
        with open(p) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def contract_stats(c):
    if not c or "tensors" not in c:
        return {}
    sizes, nbytes, ndims = [], 0, []
    for t in c["tensors"]:
        n = math.prod(t["shape"]) if t["shape"] else 1
        sizes.append(n)
        nbytes += n * DTYPE_BYTES.get(t.get("dtype", "float32"), 4)
        ndims.append(len(t["shape"]))
    total = sum(sizes) or 1
    return {
        "param_count": sum(sizes),
        "tensor_count": len(sizes),
        "param_bytes": nbytes,
        "frac_params_1d": round(sum(s for s, d in zip(sizes, ndims) if d <= 1) / total, 6),
        "frac_tensors_1d": round(sum(1 for d in ndims if d <= 1) / len(ndims), 6),
        "frac_params_2d": round(sum(s for s, d in zip(sizes, ndims) if d == 2) / total, 6),
        "frac_params_4d": round(sum(s for s, d in zip(sizes, ndims) if d == 4) / total, 6),  # conv kernels
        "frac_params_3d": round(sum(s for s, d in zip(sizes, ndims) if d == 3) / total, 6),
        "largest_tensor_frac": round(max(sizes) / total, 6),
        "dtypes": "|".join(sorted({t.get("dtype", "?") for t in c["tensors"]})),
    }


def round_metric_stats(run_dir):
    rows = []
    for p in glob.glob(os.path.join(run_dir, "client_*", "round_metrics.csv")):
        with open(p) as f:
            rows += list(csv.DictReader(f))

    def med(col):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (KeyError, ValueError, TypeError):
                pass
        return statistics.median(vals) if vals else ""

    rounds = {r.get("round") for r in rows}
    return {
        "rounds_logged": len(rounds),
        "payload_bytes": med("model_size_bytes"),
        "median_training_time_sec": med("training_time_sec"),
        "median_upload_time_sec": med("upload_time_sec"),
        "median_download_time_sec": med("download_time_sec"),
        "median_sync_wait_sec": med("sync_wait_time_sec"),
    }


def describe_run(run_dir):
    run_id = os.path.basename(run_dir.rstrip("/"))
    roles = ["server"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "client_*")))
    manifest = None
    for r in roles:
        manifest = load_json(os.path.join(run_dir, r, "role_manifest.json"))
        if manifest:
            break
    contract = load_json(first([os.path.join(run_dir, r, "model_contract.json") for r in roles]) or "")
    cfg = load_yaml(first([os.path.join(run_dir, "server", "config.yaml")] +
                          [os.path.join(run_dir, r, "config_effective.yaml") for r in roles]) or "") or {}
    h = (manifest or {}).get("hierarchy", {})
    ai, ex, fed = cfg.get("ai", {}) or {}, cfg.get("execution", {}) or {}, cfg.get("federated", {}) or {}

    client_devices = set()
    for p in glob.glob(os.path.join(run_dir, "client_*", "config_effective.yaml")):
        c = load_yaml(p) or {}
        client_devices.add((c.get("device") or {}).get("label", "?"))
    repro = (manifest or {}).get("reproducibility", {})

    pkt = glob.glob(os.path.join(run_dir, "proxy", "**", "*_packet_sequence.csv"), recursive=True)
    trace_seq = glob.glob(os.path.join(run_dir, "proxy", "**", "*__trace_*_fingerprint_sequence.csv"), recursive=True)
    pcaps = glob.glob(os.path.join(run_dir, "proxy", "**", "*.pcap*"), recursive=True)

    row = {
        "experiment_id": run_id,
        "family": h.get("family") or ai.get("family") or (contract or {}).get("family"),
        "architecture": h.get("architecture") or ai.get("architecture") or (contract or {}).get("architecture"),
        "variant": h.get("variant") or ai.get("variant") or (contract or {}).get("variant"),
        "application": h.get("application") or ai.get("application"),
        "dataset": h.get("dataset") or ai.get("dataset"),
        "framework": h.get("framework") or ai.get("framework"),
        "precision": ex.get("precision") or (contract or {}).get("precision"),
        "input_size": ai.get("input_size"), "sequence_length": ai.get("sequence_length"),
        "vocab_size": ai.get("vocab_size"), "num_classes": ai.get("num_classes"),
        "rounds_cfg": fed.get("rounds"), "local_epochs": fed.get("local_epochs"),
        "steps_per_epoch": fed.get("steps_per_epoch"), "batch_size": ex.get("batch_size"),
        "aggregation": fed.get("aggregation"), "partition": ((cfg.get("data") or {}).get("partition") or {}).get("type"),
        "server_device_label": (cfg.get("device") or {}).get("label"), "server_gpu": repro.get("gpu_name"),
        "client_devices": "|".join(sorted(client_devices)),
        "has_packet_sequence": bool(pkt), "has_trace_sequences": len(trace_seq), "has_pcap": bool(pcaps),
        "in_canonical_set": run_id in CANONICAL_RUNS,
    }
    row.update(contract_stats(contract))
    row.update(round_metric_stats(run_dir))
    if row.get("payload_bytes") and row.get("param_bytes") and row.get("tensor_count"):
        row["serialization_overhead_bytes"] = row["payload_bytes"] - row["param_bytes"]
        row["overhead_per_tensor"] = round(row["serialization_overhead_bytes"] / row["tensor_count"], 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="collected_experiments dir(s) or individual run dirs")
    ap.add_argument("--out", default="run_metadata.csv")
    a = ap.parse_args()

    run_dirs = []
    for root in a.roots:
        if os.path.basename(root.rstrip("/")).startswith("run_"):
            run_dirs.append(root)
        else:
            run_dirs += sorted(p for p in glob.glob(os.path.join(root, "run_*")) if os.path.isdir(p))
    rows = [describe_run(d) for d in run_dirs]
    if not rows:
        sys.exit("No run_* directories found.")
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    found = {r["experiment_id"] for r in rows}
    missing = [r for r in CANONICAL_RUNS if r not in found]
    print(f"Wrote {len(rows)} runs to {a.out}")
    print(f"Canonical runs found: {len(CANONICAL_RUNS) - len(missing)}/{len(CANONICAL_RUNS)}")
    for m in missing:
        print("  MISSING:", m)
    for r in rows:
        if r["in_canonical_set"]:
            print(f"  {r['experiment_id'][-8:]}  {r['family']}/{r['architecture']}/{r['variant']}  "
                  f"params={r.get('param_count')}  packet_seq={r['has_packet_sequence']}")


if __name__ == "__main__":
    main()
