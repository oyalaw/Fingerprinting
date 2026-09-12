#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default="fingerprinting_dataset/endpoint_assisted/endpoint_assisted_paired.csv",
    )
    p.add_argument(
        "--target",
        default="family",
        choices=["family", "architecture", "variant"],
    )
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.70)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="fingerprinting_dataset/endpoint_assisted/results",
    )
    return p.parse_args()

def rf(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )),
    ])

def student_model(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=1200,
            early_stopping=False,
            random_state=seed,
        )),
    ])

def align_proba(model, X, classes):
    p = model.predict_proba(X)
    have = list(model.classes_)
    out = np.zeros((len(X), len(classes)), dtype=float)
    for j, c in enumerate(classes):
        if c in have:
            out[:, j] = p[:, have.index(c)]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s

def soften(p, temperature):
    eps = 1e-9
    z = np.log(np.clip(p, eps, 1.0)) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def metrics(y_true, p, classes):
    pred = np.asarray(classes)[np.argmax(p, axis=1)]
    res = {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "macro_f1": float(
            f1_score(y_true, pred, average="macro")
        ),
        "confusion_matrix": confusion_matrix(
            y_true, pred, labels=classes
        ).tolist(),
    }
    try:
        y_onehot = pd.get_dummies(
            pd.Categorical(y_true, categories=classes)
        ).to_numpy()
        res["macro_auroc_ovr"] = float(
            roc_auc_score(
                y_onehot,
                p,
                average="macro",
                multi_class="ovr",
            )
        )
    except Exception:
        res["macro_auroc_ovr"] = None
    return res

def crossfit_teacher(Xt, y, groups, classes, n_splits, seed):
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError(
            "Need at least two independent experiments "
            "for inner teacher cross-fitting."
        )

    gkf = GroupKFold(n_splits=n_splits)
    soft = np.zeros((len(y), len(classes)), dtype=float)
    filled = np.zeros(len(y), dtype=bool)

    for k, (tr, va) in enumerate(gkf.split(Xt, y, groups)):
        model = rf(seed + 100 + k)
        model.fit(Xt.iloc[tr], y[tr])
        soft[va] = align_proba(model, Xt.iloc[va], classes)
        filled[va] = True

    if not filled.all():
        raise RuntimeError(
            "Teacher cross-fitting failed to cover "
            "every outer-train sample."
        )
    return soft

def distilled_training_set(
    X,
    y,
    teacher_p,
    classes,
    alpha,
    temperature,
):
    teacher_p = soften(teacher_p, temperature)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    hard = np.zeros_like(teacher_p)

    for i, label in enumerate(y):
        hard[i, class_to_idx[label]] = 1.0

    target = alpha * hard + (1.0 - alpha) * teacher_p

    reps = 20
    rows = []
    labels_out = []

    for i in range(len(X)):
        probs = target[i] / target[i].sum()
        counts = np.floor(probs * reps).astype(int)
        remainder = reps - counts.sum()

        if remainder > 0:
            order = np.argsort(-(probs * reps - counts))
            counts[order[:remainder]] += 1

        for j, count in enumerate(counts):
            rows.extend([i] * int(count))
            labels_out.extend([classes[j]] * int(count))

    return (
        X.iloc[rows].reset_index(drop=True),
        np.asarray(labels_out),
    )

def main():
    args = parse_args()
    df = pd.read_csv(args.data)

    target = args.target
    if target not in df.columns:
        raise RuntimeError(
            f"Target {target!r} is not present."
        )

    exp_per_class = (
        df[["experiment_id", target]]
        .drop_duplicates()
        .groupby(target)["experiment_id"]
        .nunique()
    )

    eligible = exp_per_class[
        exp_per_class >= 2
    ].index.tolist()

    excluded = exp_per_class[
        exp_per_class < 2
    ].to_dict()

    df = df[
        df[target].isin(eligible)
    ].reset_index(drop=True)

    if df.empty or len(eligible) < 2:
        raise RuntimeError(
            "Fewer than two eligible target classes remain."
        )

    teacher_cols = [
        c for c in df.columns
        if c.startswith("teacher_")
    ]
    proxy_cols = [
        c for c in df.columns
        if c.startswith("proxy_")
    ]

    if not teacher_cols or not proxy_cols:
        raise RuntimeError(
            "Paired dataset is missing teacher or proxy predictors."
        )

    y = df[target].astype(str).to_numpy()
    groups = df["experiment_id"].astype(str).to_numpy()
    classes = sorted(np.unique(y).tolist())

    n_outer = min(
        args.outer_folds,
        len(np.unique(groups)),
    )
    if n_outer < 2:
        raise RuntimeError(
            "Need at least two independent experiments."
        )

    outer = GroupKFold(n_splits=n_outer)

    oof_base = np.zeros(
        (len(df), len(classes))
    )
    oof_distill = np.zeros_like(oof_base)
    oof_teacher = np.zeros_like(oof_base)
    fold_id = np.full(len(df), -1, dtype=int)

    fold_audit = []

    for fold, (tr, te) in enumerate(
        outer.split(df, y, groups),
        start=1,
    ):
        train_groups = set(groups[tr])
        test_groups = set(groups[te])

        if train_groups & test_groups:
            raise RuntimeError(
                "Experiment leakage detected."
            )

        Xt_tr = df.iloc[tr][teacher_cols]
        Xt_te = df.iloc[te][teacher_cols]
        Xp_tr = df.iloc[tr][proxy_cols]
        Xp_te = df.iloc[te][proxy_cols]
        y_tr = y[tr]
        y_te = y[te]
        g_tr = groups[tr]

        missing = sorted(
            set(y_te) - set(y_tr)
        )
        if missing:
            raise RuntimeError(
                f"Fold {fold}: test classes absent "
                f"from training: {missing}"
            )

        # ----------------------------------------------------
        # Proxy-only baseline.
        #
        # Use exactly the same student architecture and the
        # same pseudo-sample expansion size as the distilled
        # student. The only difference is that this baseline
        # receives hard labels only.
        # ----------------------------------------------------

        uniform_teacher = np.full(
            (
                len(y_tr),
                len(classes),
            ),
            1.0 / len(classes),
            dtype=float,
        )

        X_base_aug, y_base_aug = distilled_training_set(
            Xp_tr.reset_index(
                drop=True
            ),
            y_tr,
            uniform_teacher,
            classes,
            alpha=1.0,
            temperature=1.0,
        )

        baseline = student_model(
            args.seed + fold
        )

        baseline.fit(
            X_base_aug,
            y_base_aug,
        )

        oof_base[te] = align_proba(
            baseline,
            Xp_te,
            classes,
        )

        teacher_soft_train = crossfit_teacher(
            Xt_tr.reset_index(drop=True),
            y_tr,
            g_tr,
            classes,
            args.inner_folds,
            args.seed + fold * 1000,
        )

        X_aug, y_aug = distilled_training_set(
            Xp_tr.reset_index(drop=True),
            y_tr,
            teacher_soft_train,
            classes,
            args.alpha,
            args.temperature,
        )

        distilled = student_model(
            args.seed + 10000 + fold
        )
        distilled.fit(
            X_aug,
            y_aug,
        )
        oof_distill[te] = align_proba(
            distilled,
            Xp_te,
            classes,
        )

        teacher = rf(
            args.seed + 20000 + fold
        )
        teacher.fit(
            Xt_tr,
            y_tr,
        )
        oof_teacher[te] = align_proba(
            teacher,
            Xt_te,
            classes,
        )

        fold_id[te] = fold

        fold_audit.append({
            "fold": fold,
            "train_experiments": sorted(train_groups),
            "test_experiments": sorted(test_groups),
            "train_samples": int(len(tr)),
            "test_samples": int(len(te)),
        })

    results = {
        "target": target,
        "classes": classes,
        "eligible_class_experiment_counts": {
            str(k): int(v)
            for k, v in exp_per_class.items()
            if k in eligible
        },
        "excluded_singleton_or_ineligible_classes": {
            str(k): int(v)
            for k, v in excluded.items()
        },
        "sample_count": int(len(df)),
        "experiment_count": int(
            df["experiment_id"].nunique()
        ),
        "teacher_predictor_count": len(teacher_cols),
        "proxy_predictor_count": len(proxy_cols),
        "alpha_true_label": args.alpha,
        "temperature": args.temperature,
        "baseline_proxy_student": metrics(
            y,
            oof_base,
            classes,
        ),
        "distilled_proxy_student": metrics(
            y,
            oof_distill,
            classes,
        ),
        "endpoint_teacher_upper_bound": metrics(
            y,
            oof_teacher,
            classes,
        ),
        "folds": fold_audit,
        "guardrail": (
            "Outer-test distilled student receives "
            "proxy features only."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = f"endpoint_assisted_{target}"

    (
        out_dir / f"{stem}_results.json"
    ).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    pred = df[
        ["experiment_id", "client_id", target]
    ].copy()

    pred["fold"] = fold_id

    for j, c in enumerate(classes):
        pred[f"baseline_p__{c}"] = oof_base[:, j]
        pred[f"distilled_p__{c}"] = oof_distill[:, j]
        pred[f"teacher_p__{c}"] = oof_teacher[:, j]

    pred.to_csv(
        out_dir / f"{stem}_oof_predictions.csv",
        index=False,
    )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )

if __name__ == "__main__":
    main()
