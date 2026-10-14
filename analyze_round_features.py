#!/usr/bin/env python3
"""
Analysis of the per-round passive features from extract_round_features.py.

Inputs (repo root):
    round_features/trace_summary.csv   one row per client trace
    run_metadata.csv                   labels + ground-truth model facts per experiment
    model_flops.csv                    params / training FLOPs per sample per model

Questions:
    A. Is the per-round payload recovered passively? (traffic vs logged payload)
    B. Payload size alone -> variant (closed world) and -> family (unseen model).
    C. Family from device-normalized compute features, with and without size.
    D. Size-matched band (1.5-2.6M params): can family be told apart when size can't help?
    C/D also test two fold-local refinements: per-device fixed-overhead removal (remove_overhead)
    and within-experiment cross-device gap ratios (add_device_ratios).
    E. Does the measured compute gap track training FLOPs? (validates the feature)

Grouping: every evaluation that predicts family holds out a whole *model* (not a trace or a
round), so the classifier never sees the model it is tested on. beta_vae and vae_conv build the
same network (only the loss beta differs), so they form one model group.

Device normalization uses only the training fold: for each device label, the median of the
training-fold traces is subtracted from training and held-out values of that device.

Usage:
    python3 analyze_round_features.py [--summary round_features/trace_summary.csv] [--out-dir round_features/analysis]
    python3 analyze_round_features.py --target comp_type   # classify computation type instead of family
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SAMPLES_PER_ROUND = 10          # local_epochs=1 x steps_per_epoch=10 x batch_size=1 in every labeled run
SAME_NETWORK = {"beta_vae": "vae_conv"}
BAND = (1.5e6, 2.6e6)


def computation_type(layer_types, flop_ops):
    """Computation type from the model code (model_flops.csv), independent of the application label:
    convolutional = convolutions carry > 50% of training FLOPs; recurrent = has an LSTM/GRU layer;
    attention = has attention (Transformer encoder / BERT layers); dense = Linear layers only."""
    layers = {kv.rsplit(":", 1)[0] for kv in str(layer_types).split("|") if ":" in kv}
    ops = {}
    for kv in str(flop_ops).split("|"):
        if ":" in kv:
            k, v = kv.rsplit(":", 1)
            ops[k] = float(v)
    total = sum(ops.values()) or 1.0
    if sum(v for k, v in ops.items() if "convolution" in k) / total > 0.5:
        return "convolutional"
    if layers & {"LSTM", "GRU", "RNN"}:
        return "recurrent"
    if layers & {"NonDynamicallyQuantizableLinear", "MultiheadAttention", "GELUActivation"} or \
            any("bmm" in k or "attention" in k for k in ops):
        return "attention"
    return "dense"


def load(a):
    t = pd.read_csv(a.summary)
    meta = pd.read_csv(a.metadata)
    fl = pd.read_csv(a.flops)
    fl["comp_type"] = [computation_type(l, o) for l, o in zip(fl["layer_types"], fl["flop_ops"])]
    fl = fl[["experiment_id", "params", "train_flops_per_sample", "flops_per_param", "comp_type"]]
    meta = meta[meta["family"].notna()]
    t = t.merge(meta[["experiment_id", "family", "architecture", "variant", "payload_bytes"]],
                on="experiment_id", how="inner").merge(fl, on="experiment_id", how="left")
    t["model"] = t["variant"].replace(SAME_NETWORK)
    t["app_family"] = t["family"]
    if a.target == "comp_type":   # every evaluation below classifies the "family" column
        t["family"] = t["comp_type"]
    t["exp"] = t["experiment_id"].str[-8:]
    t["payload_rel_err"] = (t["payload_up_bytes"] - t["payload_bytes"]) / t["payload_bytes"]
    # passive-only quality gate: enough rounds and a stable per-round size (no ground truth used)
    t["ok"] = (t["rounds_found"] >= 20) & (t["payload_up_cv"] < 0.05) & t["device"].notna()
    t["log_payload"] = np.log10(t["payload_up_bytes"])
    t["log_gap"] = np.log10(t["compute_gap_median"].clip(lower=1e-3))
    t["log_gap_per_MB"] = np.log10(t["compute_gap_per_MB_median"].clip(lower=1e-6))
    t["gap_iqr_rel"] = t["compute_gap_iqr"] / t["compute_gap_median"]
    t["log_up_tput"] = np.log10(t["up_throughput_median"].clip(lower=1))
    # round-to-round variability of the compute gap (std / mean over the trace's rounds)
    r = pd.read_csv(a.rounds, usecols=["experiment_id", "trace_id", "compute_gap"])
    cv = r.groupby(["experiment_id", "trace_id"])["compute_gap"].agg(lambda g: g.std() / g.mean()).rename("gap_cv")
    t = t.merge(cv.reset_index(), on=["experiment_id", "trace_id"], how="left")
    t["log_gap_cv"] = np.log10(t["gap_cv"].clip(lower=1e-4))
    return t


def device_normalize(train, test, cols):
    """Subtract per-device training-fold medians (global training median for unseen devices)."""
    tr, te = train.copy(), test.copy()
    for c in cols:
        med = tr.groupby("device")[c].median()
        g = tr[c].median()
        tr[c] = tr[c] - tr["device"].map(med)
        te[c] = te[c] - te["device"].map(med).fillna(g)
    return tr, te


GAP_CV_SET = "gap per MB + device ratios + gap CV (pre-specified)"
RATIO_REF = "dell_desktop"
RATIO_DEVS = {"jetson_orin_nano": "jet", "dell_laptop": "lap"}   # jetson_agx_orin is a different device


def remove_overhead(tr, te, q=0.1, min_n=8):
    """Per-device fixed overhead + serialization cost, fitted on the training fold only.
    Devices with fewer than min_n training traces use the all-device fit.

    linear:  the q-quantile line of gap vs payload MB (a + b*MB) is the gap of models that train
             almost instantly for their size; gap minus it estimates training compute
             -> log_compute, log_compute_per_MB.
             Payloads span 8 KB to 374 MB, so the largest models set the slope.
    log:     the q-quantile line of log10(gap) vs log10(MB): the same lower envelope, with small
             and large models given equal leverage
             -> ovr_logratio = log10(gap / floor), the excess over a size-matched floor (no clipping)
             -> log_compute_lf, log_compute_lf_per_MB from gap minus the floor."""
    from sklearn.linear_model import QuantileRegressor
    tr, te = tr.copy(), te.copy()
    for d in (tr, te):
        d["_mb"] = d["payload_up_bytes"] / 1e6
        d["_lmb"] = np.log10(d["_mb"])
        d["_lgap"] = np.log10(d["compute_gap_median"].clip(lower=1e-3))

    def per_device(x, y):
        fit = lambda d: QuantileRegressor(quantile=q, alpha=0, solver="highs").fit(d[[x]], d[y])
        g = fit(tr)
        fits = {dev: fit(sub) for dev, sub in tr.groupby("device") if len(sub) >= min_n}
        out = []
        for d in (tr, te):
            v = pd.Series(g.predict(d[[x]]), index=d.index)
            for dev, f in fits.items():
                idx = d["device"] == dev
                if idx.any():
                    v[idx] = f.predict(d.loc[idx, [x]])
            out.append(v)
        return out

    lin = per_device("_mb", "compute_gap_median")
    logf = per_device("_lmb", "_lgap")
    for d, over, lfloor in zip((tr, te), lin, logf):
        comp = (d["compute_gap_median"] - over.clip(lower=0)).clip(lower=5e-3)
        d["log_compute"] = np.log10(comp)
        d["log_compute_per_MB"] = np.log10(comp / d["_mb"])
        d["ovr_logratio"] = d["_lgap"] - lfloor
        comp_lf = (d["compute_gap_median"] - 10 ** lfloor).clip(lower=5e-3)
        d["log_compute_lf"] = np.log10(comp_lf)
        d["log_compute_lf_per_MB"] = np.log10(comp_lf / d["_mb"])
    return tr.drop(columns=["_mb", "_lmb", "_lgap"]), te.drop(columns=["_mb", "_lmb", "_lgap"])


def add_device_ratios(tr, te, col, prefix):
    """Per experiment: log10 ratio of `col` (a log10 value, so a difference) between each device in
    RATIO_DEVS and RATIO_REF. All clients of one experiment run the same model, so size and FLOPs
    cancel and what is left is how the model's layer mix runs on each device. Experiments missing
    a device get the training-fold median."""
    tr, te = tr.copy(), te.copy()
    names = []
    for d in (tr, te):
        piv = d.pivot_table(index="experiment_id", columns="device", values=col, aggfunc="median")
        for dev, short in RATIO_DEVS.items():
            name = f"{prefix}_{short}_over_desk"
            r = (piv[dev] - piv[RATIO_REF]) if dev in piv and RATIO_REF in piv else pd.Series(dtype=float)
            d[name] = d["experiment_id"].map(r)
            if name not in names:
                names.append(name)
    for n in names:
        med = tr[n].median()
        tr[n] = tr[n].fillna(med)
        te[n] = te[n].fillna(med)
    return tr, te


def prep_full(tr, te):
    """Fold-local derived features: overhead removal, then device ratios on raw gap and compute."""
    tr, te = remove_overhead(tr, te)
    tr, te = add_device_ratios(tr, te, "log_gap", "r_gap")
    tr, te = add_device_ratios(tr, te, "log_compute", "r_comp")
    tr, te = add_device_ratios(tr, te, "ovr_logratio", "r_lr")
    tr, te = add_device_ratios(tr, te, "log_compute_lf", "r_clf")
    return tr, te


def fit_predict(tr, te, feats, norm_cols=(), C=1.0, prep=None, classes=None):
    """Train on tr, return one family prediction per experiment in te (argmax of the mean class
    probabilities of its traces). `prep(train, test)` derives fold-local features first."""
    if prep is not None:
        tr, te = prep(tr, te)
    if norm_cols:
        tr, te = device_normalize(tr, te, list(norm_cols))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    clf.fit(tr[feats], tr["family"])
    P = pd.DataFrame(clf.predict_proba(te[feats]), columns=clf.classes_, index=te.index)
    P = P.reindex(columns=classes or sorted(tr["family"].unique()), fill_value=0.0)
    return [{"experiment_id": e, "model": te.loc[g.index[0], "model"], "family": te.loc[g.index[0], "family"],
             "pred": g.mean().idxmax()} for e, g in P.groupby(te["experiment_id"])]


def logo_family(t, feats, norm_cols=(), C=1.0, prep=None):
    """Leave-one-model-out family classification; balanced accuracy over experiments."""
    classes = sorted(t["family"].unique())
    preds = []
    for m in sorted(t["model"].unique()):
        preds += fit_predict(t[t["model"] != m], t[t["model"] == m], feats, norm_cols, C, prep, classes)
    p = pd.DataFrame(preds)
    return balanced_accuracy_score(p["family"], p["pred"]), p, classes


def _nested_fold(t, m, sets, classes):
    tr, te = t[t["model"] != m], t[t["model"] == m]
    inner = {name: run_set(tr, spec)[0] for name, spec in sets.items()}
    best = max(inner, key=inner.get)          # ties -> earliest (simplest) set in `sets`
    spec = sets[best]
    rows = fit_predict(tr, te, *spec[:2], prep=spec[2] if len(spec) > 2 else None, classes=classes)
    for r in rows:
        r["selected"], r["inner_ba"] = best, inner[best]
    return rows


def nested_logo(t, sets, n_jobs=-1):
    """Nested leave-one-model-out: for each held-out model, the feature set is chosen by an inner
    leave-one-model-out over the remaining models only, then scored on the held-out model.
    The outer balanced accuracy is an unbiased estimate of the whole select-then-fit procedure."""
    from joblib import Parallel, delayed
    classes = sorted(t["family"].unique())
    out = Parallel(n_jobs=n_jobs)(delayed(_nested_fold)(t, m, sets, classes) for m in sorted(t["model"].unique()))
    p = pd.DataFrame([r for rows in out for r in rows])
    return balanced_accuracy_score(p["family"], p["pred"]), p


def run_set(t, spec):
    return logo_family(t, *spec[:2], prep=spec[2] if len(spec) > 2 else None)


def slug(name):
    return (name.replace(" + ", "_and_").replace("(", "").replace(")", "").replace(",", "")
            .replace("-", "_").replace(" ", "_"))


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="round_features/trace_summary.csv")
    ap.add_argument("--rounds", default="round_features/round_features.csv")
    ap.add_argument("--metadata", default="run_metadata.csv")
    ap.add_argument("--flops", default="model_flops.csv")
    ap.add_argument("--out-dir", default=None,
                    help="default: round_features/analysis (family) or round_features/analysis_comptype")
    ap.add_argument("--exclude-models", nargs="*", default=[],
                    help="model names (variant, beta_vae merged into vae_conv) to drop before every evaluation; "
                         "a sensitivity check only, since choosing exclusions from results inflates accuracy")
    ap.add_argument("--target", choices=["family", "comp_type"], default="family",
                    help="label to classify: application family, or computation type from the model code")
    a = ap.parse_args()
    a.out_dir = a.out_dir or ("round_features/analysis" if a.target == "family" else "round_features/analysis_comptype")
    os.makedirs(a.out_dir, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_rows", 500)

    T = load(a)
    if a.exclude_models:
        unknown = set(a.exclude_models) - set(T["model"])
        if unknown:
            sys.exit(f"unknown models in --exclude-models: {sorted(unknown)}")
        T = T[~T["model"].isin(a.exclude_models)]
    t = T[T["ok"]].copy()
    T.to_csv(os.path.join(a.out_dir, "traces_annotated.csv"), index=False)

    section(f"Data (target: {a.target})")
    if a.exclude_models:
        print("EXCLUDED models (sensitivity check): " + ", ".join(a.exclude_models))
    if a.target == "comp_type":
        print(t.drop_duplicates("model").groupby(["family", "app_family"])["model"].apply(", ".join).to_string())
    print(f"labeled traces: {len(T)}  passing quality gate (>=20 rounds, payload CV<5%, device known): {len(t)}")
    bad = T[~T["ok"]]
    if len(bad):
        print("excluded traces:")
        print(bad[["exp", "variant", "trace_id", "device", "chunks", "rounds_found", "payload_up_cv"]].to_string(index=False))
    per_fam = t.groupby("family").agg(experiments=("experiment_id", "nunique"), models=("model", "nunique"), traces=("trace_id", "size"))
    print("\nper family after gate:\n" + per_fam.to_string())

    # ---------------------------------------------------------------- A
    section("A. Passive payload recovery (traffic upload bytes per round vs logged payload)")
    e = t["payload_rel_err"]
    print(f"traces: {len(t)}   |rel err| median {e.abs().median():.4%}   p90 {e.abs().quantile(.9):.4%}   max {e.abs().max():.4%}")
    print("median abs error by device (bytes):")
    print((t["payload_up_bytes"] - t["payload_bytes"]).abs().groupby(t["device"]).median().round(0).to_string())

    # ---------------------------------------------------------------- B
    section("B. Payload size alone")
    exp_pl = t.groupby(["model", "experiment_id"])["payload_up_bytes"].median().reset_index()
    mod_pl = exp_pl.groupby("model")["payload_up_bytes"].median().sort_values()
    rel_gaps = (mod_pl.diff() / mod_pl).dropna()
    within = t.groupby("experiment_id")["payload_up_bytes"].agg(lambda s: (s.max() - s.min()) / s.median())
    print(f"distinct models: {len(mod_pl)}; smallest relative gap between adjacent models' payloads: "
          f"{rel_gaps.min():.3%} ({rel_gaps.idxmin()} vs its smaller neighbour)")
    print(f"largest spread of payload across one experiment's clients: {within.max():.3%}")
    # closed world: attacker has a reference payload per known model from OTHER traces
    hit = []
    for i, r in t.iterrows():
        ref = t.drop(index=i).groupby("model")["log_payload"].median()
        hit.append(ref.sub(r["log_payload"]).abs().idxmin() == r["model"])
    print(f"closed-world model identification by nearest payload (leave-one-trace-out): "
          f"{np.mean(hit):.1%} of {len(hit)} traces")
    ba, p, classes = logo_family(t, ["log_payload"])
    print(f"open-world family from payload only (leave-one-model-out): balanced acc {ba:.3f}")
    p.to_csv(os.path.join(a.out_dir, "family_payload_only_predictions.csv"), index=False)

    # ---------------------------------------------------------------- C
    section("C. Family from device-normalized compute features (leave-one-model-out)")
    sets = {
        "payload only": (["log_payload"], []),
        "compute gap (dev-norm)": (["log_gap"], ["log_gap"]),
        "gap per MB (dev-norm)": (["log_gap_per_MB"], ["log_gap_per_MB"]),
        "gap per MB + gap variability (dev-norm)": (["log_gap_per_MB", "gap_iqr_rel"], ["log_gap_per_MB", "gap_iqr_rel"]),
        "payload + gap (dev-norm)": (["log_payload", "log_gap"], ["log_gap"]),
        "payload + gap per MB + variability": (["log_payload", "log_gap_per_MB", "gap_iqr_rel"], ["log_gap_per_MB", "gap_iqr_rel"]),
        # (2) per-device fixed overhead removed (fold-local)
        "compute (overhead-removed, dev-norm)": (["log_compute"], ["log_compute"], prep_full),
        "compute per MB (overhead-removed, dev-norm)": (["log_compute_per_MB"], ["log_compute_per_MB"], prep_full),
        # (1) cross-device ratios within an experiment
        "device ratios (raw gap)": (["r_gap_jet_over_desk", "r_gap_lap_over_desk"], [], prep_full),
        "device ratios (overhead-removed)": (["r_comp_jet_over_desk", "r_comp_lap_over_desk"], [], prep_full),
        # combinations
        "gap per MB + device ratios (raw)": (["log_gap_per_MB", "r_gap_jet_over_desk", "r_gap_lap_over_desk"],
                                             ["log_gap_per_MB"], prep_full),
        "compute per MB + device ratios (overhead-removed)": (["log_compute_per_MB", "r_comp_jet_over_desk", "r_comp_lap_over_desk"],
                                                              ["log_compute_per_MB"], prep_full),
        # log-space overhead floor (equal leverage for small and large models)
        "excess over size floor (log-space)": (["ovr_logratio"], [], prep_full),
        "compute per MB (log-floor removed, dev-norm)": (["log_compute_lf_per_MB"], ["log_compute_lf_per_MB"], prep_full),
        "device ratios (log-floor excess)": (["r_lr_jet_over_desk", "r_lr_lap_over_desk"], [], prep_full),
        "excess over size floor + device ratios (raw)": (["ovr_logratio", "r_gap_jet_over_desk", "r_gap_lap_over_desk"], [], prep_full),
        "excess over size floor + device ratios (log-floor)": (["ovr_logratio", "r_lr_jet_over_desk", "r_lr_lap_over_desk"], [], prep_full),
        "compute per MB (log-floor) + device ratios (log-floor)": (["log_compute_lf_per_MB", "r_clf_jet_over_desk", "r_clf_lap_over_desk"],
                                                                   ["log_compute_lf_per_MB"], prep_full),
        # pre-specified test (2026-10-13): baseline + round-to-round gap variability, one variant only
        GAP_CV_SET: (["log_gap_per_MB", "r_gap_jet_over_desk", "r_gap_lap_over_desk", "log_gap_cv"],
                     ["log_gap_per_MB", "log_gap_cv"], prep_full),
    }
    rows = []
    chance = 1 / t["family"].nunique()
    for name, spec in sets.items():
        ba, p, _ = run_set(t, spec)
        rows.append({"features": name, "balanced_acc": round(ba, 3), "experiments": len(p)})
        p.to_csv(os.path.join(a.out_dir, f"family_{slug(name)}_predictions.csv"), index=False)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print(f"(chance = {chance:.3f}, {t['family'].nunique()} families)")
    res.to_csv(os.path.join(a.out_dir, "family_comparison.csv"), index=False)
    best = res.loc[res["balanced_acc"].idxmax(), "features"]
    _, p, classes = run_set(t, sets[best])
    print(f"\nconfusion matrix for '{best}' (rows true, cols predicted; experiments):")
    print(pd.DataFrame(confusion_matrix(p["family"], p["pred"], labels=classes), index=classes, columns=classes).to_string())
    print(f"\nNOTE: the best of {len(sets)} sets on the same {len(p)} experiments is optimistically biased; "
          "see the nested estimate below.")

    section("C1. Pre-specified test: does round-to-round gap variability help? (one comparison)")
    base_name = "gap per MB + device ratios (raw)"
    _, pb, _ = run_set(t, sets[base_name])
    _, pc, _ = run_set(t, sets[GAP_CV_SET])
    m = pb.merge(pc, on=["experiment_id", "model", "family"], suffixes=("_base", "_cv"))
    rec = m.assign(base=m["pred_base"] == m["family"], cv=m["pred_cv"] == m["family"]).groupby("family")[["base", "cv"]].agg(["sum", "size"])
    print(f"baseline: {base_name}\n  + log gap CV (device-normalized)")
    print(f"balanced acc  baseline {balanced_accuracy_score(m['family'], m['pred_base']):.3f}   "
          f"+gap CV {balanced_accuracy_score(m['family'], m['pred_cv']):.3f}")
    print("correct per family (baseline -> +gap CV):")
    for fam in rec.index:
        print(f"  {fam:12s} {rec.loc[fam, ('base', 'sum')]}/{rec.loc[fam, ('base', 'size')]} -> {rec.loc[fam, ('cv', 'sum')]}/{rec.loc[fam, ('cv', 'size')]}")
    gained = m[(m["pred_base"] != m["family"]) & (m["pred_cv"] == m["family"])]
    lost = m[(m["pred_base"] == m["family"]) & (m["pred_cv"] != m["family"])]
    print("fixed: " + (", ".join(gained["model"]) or "none"))
    print("broken: " + (", ".join(lost["model"]) or "none"))
    print("median gap CV by family: " + ", ".join(f"{k} {v:.2f}" for k, v in t.groupby("family")["gap_cv"].median().items()))
    m.to_csv(os.path.join(a.out_dir, "family_gap_cv_test.csv"), index=False)

    section("C2. Nested leave-one-model-out (feature set chosen inside each outer fold)")
    nba, npred = nested_logo(t, sets)
    npred.to_csv(os.path.join(a.out_dir, "family_nested_predictions.csv"), index=False)
    print(f"nested balanced acc {nba:.3f}   (non-nested best {res['balanced_acc'].max():.3f}, chance {chance:.3f})")
    sel = npred.drop_duplicates("model")["selected"].value_counts()
    print("feature set chosen by the inner loop (outer folds):\n" + sel.to_string())
    print("confusion matrix (nested):")
    print(pd.DataFrame(confusion_matrix(npred["family"], npred["pred"], labels=classes), index=classes, columns=classes).to_string())
    nested_rows = [{"scope": "all", "nested_balanced_acc": round(nba, 3), "non_nested_best": res["balanced_acc"].max(),
                    "non_nested_best_set": best, "gap_per_MB": res.set_index("features").loc["gap per MB (dev-norm)", "balanced_acc"],
                    "chance": round(chance, 3), "outer_folds": npred["model"].nunique(), "candidate_sets": len(sets)}]

    # ---------------------------------------------------------------- D
    section(f"D. Size-matched band {BAND[0]/1e6:.1f}-{BAND[1]/1e6:.1f}M params")
    b = t[t["params"].between(*BAND)].copy()
    if b["family"].nunique() >= 2:
        _, bn = device_normalize(b, b, ["log_gap", "log_gap_per_MB"])
        tab = (bn.groupby(["family", "model"])
               .agg(params=("params", "first"), flops_per_param=("flops_per_param", "first"),
                    payload_MB=("payload_up_bytes", lambda s: s.median() / 1e6),
                    gap_s_raw=("compute_gap_median", "median"),
                    log_gap_dev_norm=("log_gap", "median"), traces=("trace_id", "size"))
               .sort_values(["family", "params"]))
        print(tab.round(3).to_string())
        bres = []
        for name, spec in sets.items():
            ba, p, _ = run_set(b, spec)
            bres.append({"features": name, "balanced_acc": round(ba, 3)})
        bres = pd.DataFrame(bres)
        print(f"\nleave-one-model-out ({b['model'].nunique()} models, chance {1/b['family'].nunique():.3f}):")
        print(bres.to_string(index=False))
        bres.to_csv(os.path.join(a.out_dir, "size_matched_band_comparison.csv"), index=False)
        bnba, bnp = nested_logo(b, sets)
        bnp.to_csv(os.path.join(a.out_dir, "family_nested_band_predictions.csv"), index=False)
        print(f"nested balanced acc {bnba:.3f}  (non-nested best {bres['balanced_acc'].max():.3f})")
        print("chosen sets: " + ", ".join(f"{k} x{v}" for k, v in bnp.drop_duplicates("model")["selected"].value_counts().items()))
        nested_rows.append({"scope": "size_matched_band", "nested_balanced_acc": round(bnba, 3),
                            "non_nested_best": bres["balanced_acc"].max(),
                            "non_nested_best_set": bres.loc[bres["balanced_acc"].idxmax(), "features"],
                            "gap_per_MB": bres.set_index("features").loc["gap per MB (dev-norm)", "balanced_acc"],
                            "chance": round(1 / b["family"].nunique(), 3), "outer_folds": bnp["model"].nunique(),
                            "candidate_sets": len(sets)})
        tab.to_csv(os.path.join(a.out_dir, "size_matched_band.csv"))
    else:
        print("fewer than two families in band after quality gate")

    pd.DataFrame(nested_rows).to_csv(os.path.join(a.out_dir, "family_nested.csv"), index=False)

    # ---------------------------------------------------------------- E
    section("E. Measured compute gap vs training FLOPs per round (per device)")
    t["train_flops_round"] = t["train_flops_per_sample"] * SAMPLES_PER_ROUND
    rows = []
    for dev, g in t.dropna(subset=["train_flops_round"]).groupby("device"):
        if len(g) < 5:
            continue
        r1 = spearmanr(g["train_flops_round"], g["compute_gap_median"]).correlation
        r2 = spearmanr(g["gt_training_time_median"], g["compute_gap_median"]).correlation
        x, y = np.log10(g["train_flops_round"]), np.log10(g["compute_gap_median"])
        slope = np.polyfit(x, y, 1)[0]
        over = (g["compute_gap_median"] - g["gt_training_time_median"]).median()
        rows.append({"device": dev, "traces": len(g), "spearman_gap_vs_FLOPs": round(r1, 3),
                     "loglog_slope": round(slope, 3), "spearman_gap_vs_logged_train_time": round(r2, 3),
                     "median_gap_minus_train_time_s": round(over, 3)})
    e = pd.DataFrame(rows)
    print(e.to_string(index=False))
    e.to_csv(os.path.join(a.out_dir, "gap_vs_flops.csv"), index=False)
    fam_rho = []
    for fam, g in t.dropna(subset=["train_flops_round"]).groupby("family"):
        g2 = g.copy()
        g2["lg"] = g2["log_gap"] - g2.groupby("device")["log_gap"].transform("median")
        if g2["model"].nunique() >= 3:
            fam_rho.append(f"{fam}: {spearmanr(g2['train_flops_round'], g2['lg']).correlation:.2f}")
    print("within-family Spearman (device-centred gap vs FLOPs): " + ", ".join(fam_rho))
    print(f"\nwrote {a.out_dir}/")


if __name__ == "__main__":
    main()
