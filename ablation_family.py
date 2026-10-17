#!/usr/bin/env python3
"""
Ablation of the full-trace family model (leave-one-model-out, nested selection, bootstrap CI).

Cells:
  per-client   + RF   (Fisher top-k, k nested)          per client trace (96)
  per-client   + LR   (L2, C nested)                    per client trace (96)
  cross-device + RF   (Fisher top-k, k nested)          per experiment (32)
  cross-device + LR   (L2, C nested)                    per experiment (32)   <- proposed
  cross-device + LR without compute-timing features     per experiment (32)

Usage:
    python3 ablation_family.py --out-dir fingerprinting_results/ablation_family
"""
import argparse, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ai_fingerprint.architecture_models import candidate_feature_columns

SLOT = {"dell_desktop": "desk", "dell_laptop": "lap", "jetson_orin_nano": "jet", "jetson_agx_orin": "jet"}
SAME_NET = {"beta_vae": "vae_conv"}
rng = np.random.default_rng(0)


def per_client(x_csv, y_csv, drop_compute=False):
    X = pd.read_csv(x_csv)
    Y = pd.read_csv(y_csv)[["experiment_id", "client_capture_id", "device", "family", "variant"]].drop_duplicates()
    ov = X[X["row_type"] == "overall"].merge(Y, on=["experiment_id", "client_capture_id"], how="left")
    feats = [c for c in candidate_feature_columns([c for c in X.columns if not c.startswith("_")], "size_normalized")
             if c in ov.columns and not (drop_compute and c.startswith("ct_"))]
    ov[feats] = ov[feats].fillna(ov[feats].median())
    ov["model"] = ov["variant"].replace(SAME_NET)
    ov["slot"] = ov["device"].map(SLOT)
    return ov, feats


def cross_device(ov, feats, ratios=None):
    """One vector per experiment: the three clients' features side by side, plus, when `ratios`
    is given (a DataFrame or CSV path from build_compute_features.py), the experiment-level
    cross-device ratios once. Without `ratios`, any ratio columns already in `feats` are repeated
    per client, as in the earlier single-table layout."""
    rows = []
    for exp, g in ov.groupby("experiment_id"):
        r = {"experiment_id": exp, "family": g["family"].iloc[0], "model": g["model"].iloc[0]}
        for slot in ("desk", "lap", "jet"):
            sub = g[g["slot"] == slot]
            for f in feats:
                r[f"{slot}__{f}"] = sub[f].mean() if len(sub) else np.nan
        rows.append(r)
    P = pd.DataFrame(rows)
    if ratios is not None:
        R = pd.read_csv(ratios) if isinstance(ratios, str) else ratios
        R = R.rename(columns={c: f"exp__{c}" for c in R.columns if c != "experiment_id"})
        P = P.merge(R, on="experiment_id", how="left")
    fc = [c for c in P.columns if "__" in c]
    P[fc] = P[fc].fillna(P[fc].median())
    return P, fc


def make(clf, hp, n_feat):
    if clf == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp, max_iter=5000, class_weight="balanced"))
    return make_pipeline(StandardScaler(), SelectKBest(f_classif, k=min(hp, n_feat)),
                         RandomForestClassifier(200, class_weight="balanced", n_jobs=-1, random_state=0))


GRID = {"lr": (0.03, 0.1, 0.3, 1, 3), "rf": (10, 30, 10_000)}


def nested_lomo(X, y, g, clf):
    pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        best, besthp = -1, GRID[clf][0]
        for hp in GRID[clf]:
            ip = np.empty(len(tr), dtype=object)
            for itr, ite in LeaveOneGroupOut().split(X[tr], y[tr], g[tr]):
                m = make(clf, hp, X.shape[1])
                m.fit(X[tr][itr], y[tr][itr])
                ip[ite] = m.predict(X[tr][ite])
            s = balanced_accuracy_score(y[tr], ip)
            if s > best:
                best, besthp = s, hp
        m = make(clf, besthp, X.shape[1])
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def boot_ci(y, pred, g, n=2000):
    uniq = np.unique(g)
    vals = []
    for _ in range(n):
        s = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(g == u)[0] for u in s])
        vals.append(balanced_accuracy_score(y[idx], pred[idx]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="fingerprinting_dataset/fingerprinting_X_proxy_compute2.csv")
    ap.add_argument("--y", default="fingerprinting_dataset/fingerprinting_Y_ground_truth.csv")
    ap.add_argument("--cells", nargs="+",
                    default=["per-client+LR", "cross-device+RF", "cross-device+LR (no compute features)"])
    ap.add_argument("--out-dir", default="fingerprinting_results/ablation_family")
    ap.add_argument("--ratios", default=None,
                    help="experiment_ratios.csv from build_compute_features.py; added once to the cross-device vector")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ov, feats = per_client(a.x, a.y)
    ov_nc, feats_nc = per_client(a.x, a.y, drop_compute=True)
    cells = {
        "per-client+RF": lambda: (ov[feats].values, ov, "rf"),
        "per-client+LR": lambda: (ov[feats].values, ov, "lr"),
        "cross-device+RF": lambda: (lambda P, fc: (P[fc].values, P, "rf"))(*cross_device(ov, feats, a.ratios)),
        "cross-device+LR": lambda: (lambda P, fc: (P[fc].values, P, "lr"))(*cross_device(ov, feats, a.ratios)),
        "cross-device+LR (no compute features)":
            lambda: (lambda P, fc: (P[fc].values, P, "lr"))(*cross_device(ov_nc, feats_nc)),
    }
    rows = []
    for name in a.cells:
        X, D, clf = cells[name]()
        y, g = D["family"].values, D["model"].values
        pred = nested_lomo(X.astype(float), y, g, clf)
        ba = balanced_accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro")
        lo, hi = boot_ci(y, pred, g)
        rows.append({"cell": name, "samples": len(y), "features": X.shape[1], "balanced_accuracy": round(ba, 3),
                     "macro_f1": round(f1, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
        print(f"{name:40s} n={len(y):3d} feats={X.shape[1]:4d}  bal.acc {ba:.3f}  F1 {f1:.3f}  "
              f"95% CI [{lo:.3f}, {hi:.3f}]", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "ablation_family.csv"), index=False)
    print(f"wrote {a.out_dir}/ablation_family.csv")


if __name__ == "__main__":
    main()
