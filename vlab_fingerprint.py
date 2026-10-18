#!/usr/bin/env python3
"""
Fingerprint a collected campaign end to end (used for the virtual-testbed campaign).

  1. canonicalize the proxy traces of every passing run (canonicalize_central_proxy)
  2. build X / Y (fingerprinting_dataset.build_fingerprinting_dataset)
  3. rebuild FL rounds (extract_round_features) and the compute-timing tables
     (build_compute_features: single-client features + experiment ratios)
  4. evaluate, leave-one-model-out with nested selection as in the paper:
       family        single-client RF vs cross-device LR (ratios once)
       architecture  within each family, same two methods
       variant       nearest-reference payload matching, leave-one-trace-out
       5 s windows   family, single-client RF vs cross-device LR (near-idle removed)
  5. write results.json, summary tables and confusion-matrix figures

Usage
  python3 vlab_fingerprint.py --campaign vlab_runs --out vlab_runs/analysis
"""
import argparse, csv, glob, json, os, subprocess, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from ablation_family import SAME_NET, SLOT, boot_ci, cross_device, nested_lomo, per_client
from ai_fingerprint.fingerprinting_dataset import build_fingerprinting_dataset
from canonicalize_central_proxy import canonicalize_collected_proxy


def build(campaign, out):
    runs = pd.read_csv(campaign / "runs.csv")
    # every run whose five roles all exited cleanly and whose results were collected; check_run.py
    # FAILs on strict per-run thresholds (e.g. one missed round of 15 on a tiny model) still leave
    # complete traces, and the report lists each run's check result
    rc = [c for c in runs.columns if c.startswith("rc_")]
    ok = runs[(runs[rc].astype(str) == "0").all(axis=1)]
    ids = [r for r in ok["run_id"].dropna() if (campaign / "collected" / r).exists()]
    collected = campaign / "collected"
    canon = canonicalize_collected_proxy(collected_root=collected, valid_run_ids=ids, output_dir=out / "canonical")
    feats = [Path(p) for p in canon["feature_files"]]
    gts = [Path(p) for r in ids for p in glob.glob(str(collected / r / "*" / "*ground_truth.jsonl"))]
    ds = build_fingerprinting_dataset(proxy_feature_csvs=feats, ground_truth_jsonls=gts,
                                      output_dir=out / "dataset", prefix="fingerprinting")
    sub = out / "round_subset"
    sub.mkdir(parents=True, exist_ok=True)
    for r in ids:
        link = sub / r
        if not link.exists():
            link.symlink_to(collected / r)
    subprocess.run([sys.executable, str(REPO / "extract_round_features.py"), str(sub), "--out-dir", str(out / "rounds")],
                   check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(REPO / "build_compute_features.py"), "--x", ds["X_proxy_csv"],
                    "--trace-summary", str(out / "rounds" / "trace_summary.csv"), "--out-dir", str(out / "dataset")],
                   check=True, capture_output=True, text=True)
    return ids, ds


def cm(y, p, labels):
    return confusion_matrix(y, p, labels=labels).tolist()


def evaluate(out):
    d = out / "dataset"
    x_single = d / "fingerprinting_X_proxy_single_client.csv"
    ratios = d / "fingerprinting_experiment_ratios.csv"
    y_csv = d / "fingerprinting_Y_ground_truth.csv"
    Y = pd.read_csv(y_csv)
    arch = Y[["experiment_id", "architecture"]].drop_duplicates()
    ov, feats = per_client(str(x_single), str(y_csv))
    ov = ov.merge(arch, on="experiment_id", how="left")
    P, fc = cross_device(ov, feats, str(ratios))
    P = P.merge(arch, on="experiment_id", how="left")
    fam_labels = sorted(ov["family"].unique())
    res = {"experiments": int(P.shape[0]), "client_traces": int(ov.shape[0]),
           "per_client_features": len(feats), "cross_device_features": len(fc)}

    def lomo(X, y, g, clf):
        pred = nested_lomo(X.astype(float), y, g, clf)
        lo, hi = boot_ci(y, pred, g)
        return pred, {"balanced_accuracy": round(balanced_accuracy_score(y, pred), 3),
                      "ci": [round(lo, 3), round(hi, 3)], "chance": round(1 / len(set(y)), 3), "n": int(len(y))}

    fam = {}
    p1, fam["single_client_rf"] = lomo(ov[feats].values, ov["family"].values, ov["model"].values, "rf")
    fam["single_client_rf"]["confusion"] = cm(ov["family"].values, p1, fam_labels)
    p2, fam["cross_device_lr"] = lomo(P[fc].values, P["family"].values, P["model"].values, "lr")
    fam["cross_device_lr"]["confusion"] = cm(P["family"].values, p2, fam_labels)
    fam["labels"] = fam_labels
    res["family"] = fam

    archres = {}
    for f in fam_labels:
        o, q = ov[ov["family"] == f], P[P["family"] == f]
        if o["architecture"].nunique() < 2:
            continue
        _, s = lomo(o[feats].values, o["architecture"].values, o["model"].values, "rf")
        _, c = lomo(q[fc].values, q["architecture"].values, q["model"].values, "lr")
        archres[f] = {"architectures": int(o["architecture"].nunique()), "models": int(o["model"].nunique()),
                      "single_client_rf": s, "cross_device_lr": c}
    res["architecture"] = archres

    ts = pd.read_csv(out / "rounds" / "trace_summary.csv")
    ts = ts.merge(Y[["experiment_id", "client_capture_id", "variant", "architecture", "family"]].drop_duplicates(),
                  left_on=["experiment_id", "client_id"], right_on=["experiment_id", "client_capture_id"], how="inner")
    ts["log_payload"] = np.log(ts["payload_up_bytes"].clip(lower=1))
    hits, per_arch = [], {}
    for i, r in ts.iterrows():
        ref = ts.drop(index=i).groupby("variant")["log_payload"].median()
        ok = ref.sub(r["log_payload"]).abs().idxmin() == r["variant"]
        hits.append(ok)
        per_arch.setdefault(r["architecture"], []).append(ok)
    res["variant"] = {"accuracy": round(float(np.mean(hits)), 3), "traces": len(hits),
                      "variants": int(ts["variant"].nunique()), "chance": round(1 / ts["variant"].nunique(), 3),
                      "per_architecture": {a: round(float(np.mean(v)), 3) for a, v in sorted(per_arch.items())}}

    try:
        from window_cross_device import load
        from window_denoise import cross, evaluate as wev, mark_activity, single
        W, wf = load(str(d / "fingerprinting_X_proxy.csv"), str(y_csv), 5.0)
        W = mark_activity(W)
        S, sf = single(W, wf, False)
        C, cf = cross(W, wf, True)
        sw, se = wev(S, sf, "family", "rf", 10)
        cw, ce = wev(C, cf, "family", "lr", 30)
        res["windows_5s_family"] = {"single_client_rf": {"per_window": round(sw, 3), "pooled": round(se, 3),
                                                         "windows": int(len(S))},
                                    "cross_device_lr": {"per_window": round(cw, 3), "pooled": round(ce, 3),
                                                        "windows": int(len(C))}}
    except Exception as exc:
        res["windows_5s_family"] = {"error": f"{type(exc).__name__}: {exc}"}
    return res


def figures(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = res["family"]["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, key, title in zip(axes, ("single_client_rf", "cross_device_lr"), ("Single-client RF", "Cross-device LR")):
        m = np.array(res["family"][key]["confusion"], dtype=float)
        rp = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(rp, cmap="Blues", vmin=0, vmax=1.15)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if m[i, j]:
                    ax.text(j, i, f"{100 * rp[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                            color="white" if rp[i, j] > 0.5 else "#222")
        short = [l[:3].upper() if l != "transformer" else "TF" for l in labels]
        ax.set_xticks(range(len(labels)), short)
        ax.set_yticks(range(len(labels)), short if key == "single_client_rf" else [])
        ax.set_title(f"{title} ({res['family'][key]['balanced_accuracy']:.2f})", fontsize=9)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True family")
    fig.tight_layout()
    fig.savefig(out / "family_confusion.png", dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    campaign, out = Path(a.campaign).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_build:
        ids, _ = build(campaign, out)
        print(f"built dataset from {len(ids)} runs", flush=True)
    res = evaluate(out)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    figures(res, out)
    print(json.dumps({k: v for k, v in res.items() if k != "family"}, indent=1)[:3000])
    print("family:", {k: res["family"][k]["balanced_accuracy"] for k in ("single_client_rf", "cross_device_lr")})


if __name__ == "__main__":
    main()
