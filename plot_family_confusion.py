#!/usr/bin/env python3
"""
Family confusion matrices for the paper (Fig. 2): single-client RF baseline (no shared
cross-device ratios) vs cross-device LR,
both leave-one-model-out with nested selection, on the header-corrected features.
Sized for one IEEE column (3.5 in), no suptitle, fonts readable at print size.

Usage:
    python3 plot_family_confusion.py --out paper_build/fig_family_before_after.pdf
"""
import argparse, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from ablation_family import per_client, cross_device, nested_lomo

FAM = ["autoencoder", "cnn", "rnn", "transformer"]
# run-level cross-device ratios, shared by the three clients; excluded from the single-client baseline
RATIOS = ("ct_gapratio_jet", "ct_gapratio_lap", "ct_tputratio_jet", "ct_tputratio_lap")
TICK = ["AE", "CNN", "RNN", "TF"]


def predictions(x, y, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    ov, feats = per_client(x, y)
    out = []
    single = [f for f in feats if f not in RATIOS]
    pred = nested_lomo(ov[single].values.astype(float), ov["family"].values, ov["model"].values, "rf")
    out.append(pd.DataFrame({"method": "single-client RF", "y_true": ov["family"].values, "y_pred": pred}))
    P, fc = cross_device(ov, feats)
    pred = nested_lomo(P[fc].values.astype(float), P["family"].values, P["model"].values, "lr")
    out.append(pd.DataFrame({"method": "cross-device LR", "y_true": P["family"].values, "y_pred": pred}))
    R = pd.concat(out)
    R.to_csv(cache, index=False)
    return R


def panel(ax, d, title, show_y):
    cm = confusion_matrix(d["y_true"], d["y_pred"], labels=FAM)
    rowp = cm / cm.sum(1, keepdims=True)
    ax.imshow(rowp, cmap="Blues", vmin=0, vmax=1.15)
    for i in range(4):
        for j in range(4):
            if cm[i, j]:
                ax.text(j, i, f"{100 * rowp[i, j]:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if rowp[i, j] > 0.5 else "#222222")
    ax.set_xticks(np.arange(-0.5, 4), minor=True); ax.set_yticks(np.arange(-0.5, 4), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(TICK, fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(TICK if show_y else [], fontsize=8)
    if show_y:
        ax.set_ylabel("True family", fontsize=8)
    ba = balanced_accuracy_score(d["y_true"], d["y_pred"])
    ax.set_xlabel(f"Predicted\n\n{title} ({ba:.2f})", fontsize=8, linespacing=1.0)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="fingerprinting_dataset/fingerprinting_X_proxy_compute3.csv")
    ap.add_argument("--y", default="fingerprinting_dataset/fingerprinting_Y_ground_truth.csv")
    ap.add_argument("--cache", default="fingerprinting_results/ratio_checks/family_lomo_predictions_single_client.csv")
    ap.add_argument("--out", default="paper_build/fig_family_before_after.pdf")
    a = ap.parse_args()
    R = predictions(a.x, a.y, a.cache)
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    panel(axes[0], R[R["method"] == "single-client RF"], "(a) Single-client RF", True)
    panel(axes[1], R[R["method"] == "cross-device LR"], "(b) Cross-device LR", False)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.25, wspace=0.12)
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=300)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
