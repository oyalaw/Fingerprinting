#!/usr/bin/env python3
"""
Figures for the per-round passive feature analysis (run analyze_round_features.py first).

Reads round_features/analysis/ and writes PNG + PDF figures to round_features/analysis/figures/:
    fig1_payload_by_family      model payload size vs family: size identifies the model, not the family
    fig2_family_accuracy        leave-one-model-out family balanced accuracy per feature set
    fig3_size_matched_band      device-normalized compute gap for size-matched models (1.5-2.6M params)
    fig4_gap_vs_flops           measured compute gap vs training FLOPs per round, one panel per device
    fig5_confusion_best         family confusion matrix for the best feature set
    fig6_nested_vs_non_nested   selection bias: best-of-N feature set vs nested selection
    fig7_confusion_nested       family confusion matrices under nested selection: all models | size-matched band

Usage:
    python3 plot_round_features.py [--analysis-dir round_features/analysis]
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analyze_round_features import slug

# Reference categorical slots 1-3 + violet: this 4-set clears the all-pairs normal-vision (>=15)
# and CVD (>=8) floors; slot 4 (yellow) fails normal vision against orange in scatters.
FAMILY_COLOR = {"autoencoder": "#2a78d6", "cnn": "#eb6834", "rnn": "#1baf7a", "transformer": "#4a3aa7"}
FAMILY_MARKER = {"autoencoder": "o", "cnn": "s", "rnn": "^", "transformer": "D"}
FAMILY_LABEL = {"autoencoder": "Autoencoder", "cnn": "CNN", "rnn": "RNN", "transformer": "Transformer"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SAMPLES_PER_ROUND = 10

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def family_legend(ax, fams, **kw):
    h = [plt.Line2D([], [], ls="", marker=FAMILY_MARKER[f], ms=7, mfc=FAMILY_COLOR[f], mec=SURFACE,
                    mew=1.2, label=FAMILY_LABEL[f]) for f in fams]
    ax.legend(handles=h, **kw)


def scatter(ax, d, x, y, s=46):
    for f, g in d.groupby("family"):
        ax.scatter(g[x], g[y], s=s, marker=FAMILY_MARKER[f], c=FAMILY_COLOR[f],
                   edgecolors=SURFACE, linewidths=1.2, zorder=3)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig1(t, out):
    m = (t.groupby(["family", "model"]).agg(payload=("payload_up_bytes", "median")).reset_index())
    fams = sorted(m["family"].unique())
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    rng = np.random.default_rng(0)
    for i, f in enumerate(fams):
        g = m[m["family"] == f]
        ax.scatter(g["payload"], np.full(len(g), i) + rng.uniform(-0.12, 0.12, len(g)), s=46,
                   marker=FAMILY_MARKER[f], c=FAMILY_COLOR[f], edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(range(len(fams)), [f"{FAMILY_LABEL[f]} ({(m['family'] == f).sum()})" for f in fams])
    ax.set_ylim(-0.6, len(fams) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Per-round upload payload recovered from traffic (bytes, log scale)")
    ax.set_title("Each model has a distinct payload, but family size ranges overlap", loc="left")
    save(fig, out, "fig1_payload_by_family")


def fig2(res, out, chance):
    r = res.set_index("features").iloc[::-1]  # analysis order, first set on top
    best = r["balanced_acc"][::-1].idxmax()  # ties go to the simpler set (earlier in order)
    fig, ax = plt.subplots(figsize=(6.6, 0.8 + 0.34 * len(r)))
    y = np.arange(len(r))
    ax.barh(y, r["balanced_acc"], height=0.62,
            color=[SEQ_BLUE[4] if i == best else SEQ_BLUE[2] for i in r.index], zorder=3)
    for yi, v in zip(y, r["balanced_acc"]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", color=INK, fontsize=8.5)
    ax.axvline(chance, color=INK2, lw=1, ls=(0, (3, 2)), zorder=4)
    ax.text(chance + 0.008, -0.55, f"chance {chance:.2f}", color=INK2, fontsize=8, va="center")
    ax.set_yticks(y, r.index)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.85, len(r) - 0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Family balanced accuracy (leave-one-model-out, per experiment)")
    ax.set_title("Compute per byte plus cross-device ratios gives the best family accuracy", loc="left")
    save(fig, out, "fig2_family_accuracy")


def fig3(t, out, band=(1.5e6, 2.6e6)):
    b = t[t["params"].between(*band)].copy()
    b["lg"] = b["log_gap"] - b.groupby("device")["log_gap"].transform("median")
    m = b.groupby(["family", "model"]).agg(params=("params", "first"), lg=("lg", "median"),
                                           lo=("lg", "min"), hi=("lg", "max")).reset_index()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for _, r in m.iterrows():
        ax.plot([r["params"] / 1e6] * 2, [r["lo"], r["hi"]], color=FAMILY_COLOR[r["family"]],
                lw=2, alpha=0.45, solid_capstyle="round", zorder=2)
    m["px"] = m["params"] / 1e6
    scatter(ax, m, "px", "lg", s=60)
    nudge = {"bert_tiny": (8, -11), "tiny_transformer_2layer": (8, 13), "vae_fc": (-7, 0)}  # crowded 1.6-1.7M corner
    for _, r in m.iterrows():
        dx, dy = nudge.get(r["model"], (6, 0))
        ax.annotate(r["model"].replace("_", " "), (r["px"], r["lg"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, color=INK2, va="center",
                    ha="right" if dx < 0 else "left")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Compute gap, log10 vs device median")
    ax.set_xlim(band[0] / 1e6 - 0.05, band[1] / 1e6 + 0.45)
    ax.set_title("At matched size, CNNs show the longest compute gap", loc="left")
    family_legend(ax, sorted(m["family"].unique()), loc="upper left", fontsize=8)
    save(fig, out, "fig3_size_matched_band")


def fig4(t, out):
    d = t.dropna(subset=["train_flops_per_sample"]).copy()
    d["flops_round"] = d["train_flops_per_sample"] * SAMPLES_PER_ROUND
    gv = pd.read_csv(os.path.join(os.path.dirname(out), "gap_vs_flops.csv")).set_index("device")
    devs = [x for x in ["dell_desktop", "dell_laptop", "jetson_orin_nano"] if x in set(d["device"])]
    fig, axes = plt.subplots(1, len(devs), figsize=(7.2, 2.9), sharex=True, sharey=True)
    for ax, dev in zip(np.atleast_1d(axes), devs):
        g = d[d["device"] == dev]
        scatter(ax, g, "flops_round", "compute_gap_median", s=34)
        ax.set_xscale("log"); ax.set_yscale("log")
        rho = gv.loc[dev, "spearman_gap_vs_FLOPs"] if dev in gv.index else np.nan
        ax.set_title(f"{dev.replace('_', ' ')}  (ρ = {rho:.2f}, n = {len(g)})", fontsize=8.5, loc="left")
        ax.set_xlabel("Training FLOPs per round")
    np.atleast_1d(axes)[0].set_ylabel("Measured compute gap (s)")
    family_legend(np.atleast_1d(axes)[0], sorted(d["family"].unique()), loc="upper left", fontsize=7)
    fig.suptitle("The passive compute gap tracks training FLOPs on every device", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig4_gap_vs_flops")


def fig5(adir, out, res):
    best = res.loc[res["balanced_acc"].idxmax(), "features"]
    p = pd.read_csv(os.path.join(adir, f"family_{slug(best)}_predictions.csv"))
    cls = sorted(set(p["family"]) | set(p["pred"]))
    cm = pd.crosstab(p["family"], p["pred"]).reindex(index=cls, columns=cls, fill_value=0)
    frac = cm.div(cm.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.imshow(frac.values, cmap=LinearSegmentedColormap.from_list("seq", SEQ_BLUE), vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            v = frac.values[i, j]
            ax.text(j, i, f"{cm.values[i, j]}\n{v:.0%}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.55 else INK)
    lab = [FAMILY_LABEL[c] for c in cls]
    ax.set_xticks(range(len(cls)), lab, rotation=30, ha="right")
    ax.set_yticks(range(len(cls)), lab)
    ax.set_xlabel("Predicted family"); ax.set_ylabel("True family")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{best}\n(experiments; row %)", loc="left", fontsize=8.5)
    save(fig, out, "fig5_confusion_best")


def fig6(adir, out):
    n = pd.read_csv(os.path.join(adir, "family_nested.csv")).set_index("scope")
    scopes = [s for s in ["all", "size_matched_band"] if s in n.index]
    title = {"all": "All models", "size_matched_band": "Size-matched, 1.5-2.6M params"}
    fig, axes = plt.subplots(1, len(scopes), figsize=(7.4, 2.5), sharex=True, squeeze=False)
    for ax, sc in zip(axes[0], scopes):
        r = n.loc[sc]
        rows = [(f"Best of {int(r['candidate_sets'])} sets, not nested", r["non_nested_best"], SEQ_BLUE[2]),
                ("Gap per MB, single set", r["gap_per_MB"], SEQ_BLUE[2]),
                ("Nested selection", r["nested_balanced_acc"], SEQ_BLUE[5])]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, [v for _, v, _ in rows], height=0.6, color=[c for _, _, c in rows], zorder=3)
        for yi, (_, v, _) in zip(y, rows):
            ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        ax.axvline(r["chance"], color=INK2, lw=1, ls=(0, (3, 2)), zorder=4)
        ax.text(r["chance"] + 0.012, -0.62, f"chance {r['chance']:.2f}", fontsize=7.5, color=INK2, va="center")
        ax.set_ylim(-0.9, len(rows) - 0.45)
        ax.set_yticks(y, [lab for lab, _, _ in rows] if ax is axes[0][0] else [""] * len(rows))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1)
        ax.set_title(f"{title[sc]}  ({int(r['outer_folds'])} models)", loc="left", fontsize=9)
        ax.set_xlabel("Family balanced accuracy")
    fig.suptitle("Choosing features inside the CV lowers the all-model estimate; the size-matched result holds",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig6_nested_vs_non_nested")


def confusion_panel(ax, p, title):
    cls = sorted(set(p["family"]) | set(p["pred"]))
    cm = pd.crosstab(p["family"], p["pred"]).reindex(index=cls, columns=cls, fill_value=0)
    frac = cm.div(cm.sum(axis=1), axis=0)
    ax.imshow(frac.values, cmap=LinearSegmentedColormap.from_list("seq", SEQ_BLUE), vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            v = frac.values[i, j]
            ax.text(j, i, f"{cm.values[i, j]}\n{v:.0%}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.55 else INK)
    lab = [FAMILY_LABEL[c] for c in cls]
    ax.set_xticks(range(len(cls)), lab, rotation=30, ha="right")
    ax.set_yticks(range(len(cls)), [f"{l} ({n})" for l, n in zip(lab, cm.sum(axis=1))])
    ax.set_xlabel("Predicted family")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    correct = int(np.trace(cm.values))
    ax.set_title(f"{title}\n{correct} of {int(cm.values.sum())} experiments correct", loc="left", fontsize=9)


def fig7(adir, out):
    panels = [("family_nested_predictions.csv", "All models (chance 25%)"),
              ("family_nested_band_predictions.csv", "Same size, 1.5-2.6M params (chance 33%)")]
    panels = [(f, t) for f, t in panels if os.path.exists(os.path.join(adir, f))]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.9),
                             gridspec_kw={"width_ratios": [4, 3][:len(panels)]}, squeeze=False)
    for ax, (f, t) in zip(axes[0], panels):
        confusion_panel(ax, pd.read_csv(os.path.join(adir, f)), t)
    axes[0][0].set_ylabel("True family (experiments)")
    fig.suptitle("Family of an unseen model, nested leave-one-model-out (count and row %)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig7_confusion_nested")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="round_features/analysis")
    a = ap.parse_args()
    out = os.path.join(a.analysis_dir, "figures")
    os.makedirs(out, exist_ok=True)
    t = pd.read_csv(os.path.join(a.analysis_dir, "traces_annotated.csv"))
    t = t[t["ok"]]
    res = pd.read_csv(os.path.join(a.analysis_dir, "family_comparison.csv"))
    fig1(t, out)
    fig2(res, out, 1 / t["family"].nunique())
    fig3(t, out)
    fig4(t, out)
    fig5(a.analysis_dir, out, res)
    fig6(a.analysis_dir, out)
    fig7(a.analysis_dir, out)


if __name__ == "__main__":
    main()
