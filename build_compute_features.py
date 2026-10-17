#!/usr/bin/env python3
"""
Build the compute-timing features from the round reconstruction, keeping single-client and
cross-device information apart.

Inputs
  fingerprinting_X_proxy.csv     per-client traffic features (68, size-normalized)
  trace_summary.csv              per-trace round statistics from extract_round_features.py

Outputs
  X_proxy_single_client.csv      X + five compute-timing features computed from each client's own
                                 trace only (strictly single-client):
                                   ct_log_gap_per_mb, ct_gap_cv, ct_gap_p90_p10,
                                   ct_roundperiod_over_gap, ct_wait_frac
  experiment_ratios.csv          one row per experiment with the four cross-device ratios, which
                                 compare clients and so belong to the experiment, not to a client:
                                   ct_gapratio_jet, ct_gapratio_lap   log10 compute gap vs desktop
                                   ct_tputratio_jet, ct_tputratio_lap log10 upload rate vs desktop
  X_proxy_compute_legacy.csv     (--legacy) the previous single-table layout with the ratios
                                 copied into every client row, for reproducing earlier results

Missing values of the overall (complete-trace) rows are filled with the column median, as before.
ablation_family.per_client() reads the single-client table; ablation_family.cross_device() adds
the experiment ratios once (--ratios).

Usage
  python3 extract_round_features.py collected_experiments --out-dir round_features_v2
  python3 build_compute_features.py --trace-summary round_features_v2/trace_summary.csv
"""
import argparse, os
import numpy as np
import pandas as pd

SINGLE = ["ct_log_gap_per_mb", "ct_gap_cv", "ct_gap_p90_p10", "ct_roundperiod_over_gap", "ct_wait_frac"]
RATIOS = ["ct_gapratio_jet", "ct_gapratio_lap", "ct_tputratio_jet", "ct_tputratio_lap"]
REF = "dell_desktop"
# Only the Orin Nano defines the Jetson ratio, as in the features used for the paper; the two runs
# with an AGX Orin client get the median.
OTHERS = (("jetson_orin_nano", "jet"), ("dell_laptop", "lap"))


def single_client(ts):
    f = ts[["experiment_id", "client_id"]].rename(columns={"client_id": "client_capture_id"}).copy()
    f["ct_log_gap_per_mb"] = np.log10(ts["compute_gap_per_MB_median"].clip(lower=1e-6))
    f["ct_gap_cv"] = ts["compute_gap_iqr"] / ts["compute_gap_median"].replace(0, np.nan)
    f["ct_gap_p90_p10"] = np.log10((ts["compute_gap_p90"] / ts["compute_gap_p10"].replace(0, np.nan)).clip(lower=1e-6))
    f["ct_roundperiod_over_gap"] = np.log10(
        (ts["round_period_median"] / ts["compute_gap_median"].replace(0, np.nan)).clip(lower=1e-6))
    f["ct_wait_frac"] = ts["wait_after_upload_median"] / ts["round_period_median"].replace(0, np.nan)
    return f


def experiment_ratios(ts):
    """log10 ratio of each device's median to the desktop's, per experiment."""
    out = pd.DataFrame(index=sorted(ts["experiment_id"].unique()))
    for col, tag in (("compute_gap_median", "ct_gapratio"), ("up_throughput_median", "ct_tputratio")):
        piv = ts.pivot_table(index="experiment_id", columns="device", values=col, aggfunc="median")
        for dev, slot in OTHERS:
            name = f"{tag}_{slot}"
            if dev in piv and REF in piv:
                val = np.log10((piv[dev] / piv[REF]).clip(lower=1e-6))
                out[name] = out[name].fillna(val) if name in out else val
            elif name not in out:
                out[name] = np.nan
    out.index.name = "experiment_id"
    return out[RATIOS].reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--x", default="fingerprinting_dataset/fingerprinting_X_proxy.csv")
    ap.add_argument("--trace-summary", default="round_features_v2/trace_summary.csv")
    ap.add_argument("--out-dir", default="fingerprinting_dataset")
    ap.add_argument("--legacy", action="store_true", help="also write the previous combined layout")
    a = ap.parse_args()

    X = pd.read_csv(a.x)
    ts = pd.read_csv(a.trace_summary)
    ov = X["row_type"] == "overall"

    S = X.merge(single_client(ts), on=["experiment_id", "client_capture_id"], how="left")
    for c in SINGLE:
        S.loc[ov, c] = S.loc[ov, c].fillna(S.loc[ov, c].median())
    R = experiment_ratios(ts)
    R = R[R["experiment_id"].isin(set(X["experiment_id"]))]
    for c in RATIOS:
        R[c] = R[c].fillna(R[c].median())

    os.makedirs(a.out_dir, exist_ok=True)
    p_single = os.path.join(a.out_dir, "fingerprinting_X_proxy_single_client.csv")
    p_ratio = os.path.join(a.out_dir, "fingerprinting_experiment_ratios.csv")
    S.to_csv(p_single, index=False)
    R.to_csv(p_ratio, index=False)
    print(f"wrote {p_single} ({len(SINGLE)} single-client compute features) and {p_ratio} "
          f"({len(R)} experiments x {len(RATIOS)} ratios)")
    if a.legacy:
        L = S.merge(R, on="experiment_id", how="left")
        p_leg = os.path.join(a.out_dir, "fingerprinting_X_proxy_compute_legacy.csv")
        L.to_csv(p_leg, index=False)
        print(f"wrote {p_leg} (ratios copied into every client row, as in earlier results)")


if __name__ == "__main__":
    main()
