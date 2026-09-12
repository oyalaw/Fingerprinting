#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


NONTRANSFERABLE_PERFORMANCE_TOKENS = (
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "_tp__",
    "_tn__",
    "_fp__",
    "_fn__",
)


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
    p.add_argument(
        "--teacher-profile",
        default="transferable",
        choices=["transferable", "full"],
    )
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--alphas", default="1.0,0.9,0.8,0.7,0.5")
    p.add_argument("--temperatures", default="1.0,2.0,4.0")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden1", type=int, default=128)
    p.add_argument("--hidden2", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        default="fingerprinting_dataset/endpoint_assisted/nested_results",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_grid(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty numeric grid.")
    return vals


def teacher_columns_for_profile(columns: Sequence[str], profile: str) -> List[str]:
    teacher = [c for c in columns if c.startswith("teacher_")]
    if profile == "full":
        return teacher
    return [
        c for c in teacher
        if not any(tok in c.lower() for tok in NONTRANSFERABLE_PERFORMANCE_TOKENS)
    ]


def proxy_columns(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if c.startswith("proxy_")]



def experiment_stratified_splits(
    y,
    groups,
    desired_splits,
    seed,
):
    """Create stratified folds at the independent experiment level.

    Each experiment receives exactly one family label.  Splitting is
    performed on unique experiments first; the resulting experiment
    membership is then expanded back to client-trace row indices.
    """

    y = np.asarray(y).astype(str)
    groups = np.asarray(groups).astype(str)

    meta = pd.DataFrame(
        {
            "experiment_id": groups,
            "label": y,
        }
    ).drop_duplicates()

    label_count_per_experiment = (
        meta.groupby("experiment_id")["label"]
        .nunique()
    )

    if (
        label_count_per_experiment.max()
        > 1
    ):
        raise RuntimeError(
            "An experiment contains multiple target labels."
        )

    counts = (
        meta.groupby("label")[
            "experiment_id"
        ]
        .nunique()
    )

    n_splits = min(
        int(desired_splits),
        int(counts.min()),
    )

    if n_splits < 2:
        raise RuntimeError(
            "Insufficient independent experiments "
            "per class for stratified CV."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    exp_ids = (
        meta["experiment_id"]
        .astype(str)
        .to_numpy()
    )

    exp_labels = (
        meta["label"]
        .astype(str)
        .to_numpy()
    )

    dummy = np.zeros(
        len(meta),
        dtype=np.int8,
    )

    splits = []

    for exp_tr, exp_va in skf.split(
        dummy,
        exp_labels,
    ):
        train_experiments = set(
            exp_ids[exp_tr]
        )

        valid_experiments = set(
            exp_ids[exp_va]
        )

        tr = np.flatnonzero(
            np.isin(
                groups,
                list(train_experiments),
            )
        )

        va = np.flatnonzero(
            np.isin(
                groups,
                list(valid_experiments),
            )
        )

        # Every validation split must represent
        # every available class.
        expected = set(
            exp_labels
        )

        actual = set(
            y[va]
        )

        if actual != expected:
            raise RuntimeError(
                "Experiment-level stratification failed "
                f"to preserve all classes: "
                f"expected={sorted(expected)}, "
                f"actual={sorted(actual)}"
            )

        if (
            set(groups[tr])
            & set(groups[va])
        ):
            raise RuntimeError(
                "Experiment leakage detected."
            )

        splits.append(
            (
                tr,
                va,
            )
        )

    return splits


def rf_teacher(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


@dataclass
class MatrixPreprocessor:
    imputer: SimpleImputer
    scaler: StandardScaler

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "MatrixPreprocessor":
        imp = SimpleImputer(strategy="median")
        Xi = imp.fit_transform(X)
        sc = StandardScaler()
        sc.fit(Xi)
        return cls(imp, sc)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xi = self.imputer.transform(X)
        Xs = self.scaler.transform(Xi)
        return np.asarray(Xs, dtype=np.float32)


class StudentMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden1: int, hidden2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def soften_probabilities(p: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0)
    z = np.log(p) / float(temperature)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_student(
    X_train: pd.DataFrame,
    y_train_idx: np.ndarray,
    n_classes: int,
    *,
    teacher_probs: np.ndarray | None,
    alpha: float,
    temperature: float,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden1: int,
    hidden2: int,
    dropout: float,
):
    set_seed(seed)

    prep = MatrixPreprocessor.fit(X_train)
    X_np = prep.transform(X_train)
    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(np.asarray(y_train_idx, dtype=np.int64))

    model = StudentMLP(
        X_np.shape[1],
        n_classes,
        hidden1,
        hidden2,
        dropout,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    q_t = None
    if teacher_probs is not None:
        q_np = soften_probabilities(teacher_probs, temperature).astype(np.float32)
        q_t = torch.from_numpy(q_np)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        hard_loss = F.cross_entropy(logits, y_t)

        if alpha >= 1.0 or q_t is None:
            loss = hard_loss
        else:
            log_p_t = F.log_softmax(logits / temperature, dim=1)
            soft_loss = F.kl_div(
                log_p_t,
                q_t,
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = alpha * hard_loss + (1.0 - alpha) * soft_loss

        loss.backward()
        optimizer.step()

    return prep, model


def predict_student(prep: MatrixPreprocessor, model: StudentMLP, X: pd.DataFrame) -> np.ndarray:
    X_np = prep.transform(X)
    X_t = torch.from_numpy(X_np)
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(X_t), dim=1).cpu().numpy()
    return np.asarray(p, dtype=np.float64)


def align_teacher_proba(model, X: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    p = model.predict_proba(X)
    model_classes = list(model.classes_)
    out = np.zeros((len(X), len(classes)), dtype=np.float64)

    for j, c in enumerate(classes):
        if c in model_classes:
            out[:, j] = p[:, model_classes.index(c)]

    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def crossfit_teacher_probs(
    X_teacher: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    classes: Sequence[str],
    *,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    teacher_splits = experiment_stratified_splits(
        y,
        groups,
        n_splits,
        seed,
    )
    out = np.zeros((len(y), len(classes)), dtype=np.float64)
    filled = np.zeros(len(y), dtype=bool)

    for fold, (tr, va) in enumerate(
        teacher_splits,
        start=1,
    ):
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_teacher.iloc[tr])
        Xva = imp.transform(X_teacher.iloc[va])

        teacher = rf_teacher(seed + fold)
        teacher.fit(Xtr, y[tr])

        out[va] = align_teacher_proba(teacher, Xva, classes)
        filled[va] = True

    if not filled.all():
        raise RuntimeError("Teacher cross-fitting did not cover all samples.")

    return out


def probability_metrics(y_true: np.ndarray, p: np.ndarray, classes: Sequence[str]) -> Dict:
    classes_arr = np.asarray(classes)
    pred = classes_arr[np.argmax(p, axis=1)]

    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=classes).tolist(),
        "per_class_recall": {},
    }

    cm = np.asarray(result["confusion_matrix"], dtype=float)
    for i, c in enumerate(classes):
        denom = cm[i].sum()
        result["per_class_recall"][c] = float(cm[i, i] / denom) if denom > 0 else None

    try:
        if set(y_true) != set(classes):
            raise ValueError(
                "AUROC requires all evaluated classes "
                "in this validation set."
            )

        y_onehot = np.zeros(
            (len(y_true), len(classes)),
            dtype=int,
        )
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for i, label in enumerate(y_true):
            y_onehot[i, class_to_idx[label]] = 1

        result["macro_auroc_ovr"] = float(
            roc_auc_score(
                y_onehot,
                p,
                average="macro",
                multi_class="ovr",
            )
        )
    except Exception:
        result["macro_auroc_ovr"] = None

    return result


def fuse_by_experiment(df_meta: pd.DataFrame, y: np.ndarray, p: np.ndarray):
    tmp = df_meta[["experiment_id"]].copy()
    tmp["label"] = y

    for j in range(p.shape[1]):
        tmp[f"p{j}"] = p[:, j]

    y_out = []
    p_out = []
    exp_out = []

    for experiment_id, sub in tmp.groupby("experiment_id", sort=True):
        labels = sub["label"].unique()
        if len(labels) != 1:
            raise RuntimeError(
                f"Experiment {experiment_id} contains multiple target labels."
            )

        y_out.append(labels[0])
        p_out.append(
            sub[[f"p{j}" for j in range(p.shape[1])]]
            .mean()
            .to_numpy(dtype=float)
        )
        exp_out.append(experiment_id)

    return np.asarray(y_out), np.vstack(p_out), exp_out


def candidate_grid(alphas, temperatures):
    out = []
    for alpha in alphas:
        if alpha >= 1.0:
            out.append((1.0, 1.0))
        else:
            for temperature in temperatures:
                out.append((float(alpha), float(temperature)))

    seen = set()
    unique = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def select_distillation_hyperparameters(
    X_proxy,
    X_teacher,
    y,
    groups,
    classes,
    *,
    candidates,
    inner_folds,
    seed,
    epochs,
    lr,
    weight_decay,
    hidden1,
    hidden2,
    dropout,
):
    inner_splits = experiment_stratified_splits(
        y,
        groups,
        inner_folds,
        seed,
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    candidate_results = []

    for candidate_index, (alpha, temperature) in enumerate(candidates):
        bal_scores = []
        f1_scores = []

        for inner_fold, (tr, va) in enumerate(
            inner_splits,
            start=1,
        ):
            if set(y[va]) != set(classes):
                raise RuntimeError(
                    "Inner validation fold does not "
                    "contain every target class."
                )

            if set(y[va]) - set(y[tr]):
                raise RuntimeError(
                    "Inner validation contains a class "
                    "absent from inner training."
                )

            teacher_soft = None
            if alpha < 1.0:
                teacher_soft = crossfit_teacher_probs(
                    X_teacher.iloc[tr].reset_index(drop=True),
                    y[tr],
                    groups[tr],
                    classes,
                    n_splits=min(inner_folds, len(np.unique(groups[tr]))),
                    seed=seed + inner_fold * 100,
                )

            y_tr_idx = np.asarray(
                [class_to_idx[label] for label in y[tr]],
                dtype=np.int64,
            )

            prep, student = train_student(
                X_proxy.iloc[tr].reset_index(drop=True),
                y_tr_idx,
                len(classes),
                teacher_probs=teacher_soft,
                alpha=alpha,
                temperature=temperature,
                seed=seed + inner_fold,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                hidden1=hidden1,
                hidden2=hidden2,
                dropout=dropout,
            )

            p_val = predict_student(prep, student, X_proxy.iloc[va])
            # Fuse client-trace probabilities within each
            # independent experiment before model selection.
            y_val_exp, p_val_exp, _ = fuse_by_experiment(
                pd.DataFrame(
                    {
                        "experiment_id":
                            groups[va]
                    }
                ),
                y[va],
                p_val,
            )

            m = probability_metrics(
                y_val_exp,
                p_val_exp,
                classes,
            )

            bal_scores.append(
                m["balanced_accuracy"]
            )

            f1_scores.append(
                m["macro_f1"]
            )

        if bal_scores:
            candidate_results.append({
                "alpha": float(alpha),
                "temperature": float(temperature),
                "mean_inner_balanced_accuracy": float(np.mean(bal_scores)),
                "mean_inner_macro_f1": float(np.mean(f1_scores)),
                "scored_inner_folds": int(len(bal_scores)),
            })

    if not candidate_results:
        raise RuntimeError("No distillation candidate could be evaluated.")

    best = max(
        candidate_results,
        key=lambda r: (
            r["mean_inner_balanced_accuracy"],
            r["mean_inner_macro_f1"],
            r["alpha"],
        ),
    )

    return (
        (float(best["alpha"]), float(best["temperature"])),
        candidate_results,
    )


def main():
    args = parse_args()

    df = pd.read_csv(args.data)
    target = args.target

    if target not in df.columns:
        raise RuntimeError(f"Target {target!r} not present.")

    teacher_cols = teacher_columns_for_profile(df.columns, args.teacher_profile)
    proxy_cols = proxy_columns(df.columns)

    if not teacher_cols:
        raise RuntimeError("No teacher predictors remain.")
    if not proxy_cols:
        raise RuntimeError("No proxy predictors found.")

    exp_per_class = (
        df[["experiment_id", target]]
        .drop_duplicates()
        .groupby(target)["experiment_id"]
        .nunique()
    )

    eligible = exp_per_class[exp_per_class >= 2].index.tolist()
    excluded = exp_per_class[exp_per_class < 2].to_dict()

    df = df[df[target].isin(eligible)].reset_index(drop=True)

    if len(eligible) < 2:
        raise RuntimeError("Need >=2 eligible target classes.")

    y = df[target].astype(str).to_numpy()
    groups = df["experiment_id"].astype(str).to_numpy()
    classes = sorted(np.unique(y).tolist())

    Xp = df[proxy_cols]
    Xt = df[teacher_cols]

    alphas = parse_float_grid(args.alphas)
    temperatures = parse_float_grid(args.temperatures)
    candidates = candidate_grid(alphas, temperatures)

    outer_splits = experiment_stratified_splits(
        y,
        groups,
        args.outer_folds,
        args.seed,
    )

    oof_baseline = np.zeros((len(df), len(classes)), dtype=float)
    oof_distilled = np.zeros_like(oof_baseline)
    oof_teacher = np.zeros_like(oof_baseline)
    fold_id = np.full(len(df), -1, dtype=int)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    fold_records = []

    for outer_fold, (tr, te) in enumerate(
        outer_splits,
        start=1,
    ):
        train_groups = set(groups[tr])
        test_groups = set(groups[te])

        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"Outer experiment leakage: {sorted(overlap)}")

        missing = set(y[te]) - set(y[tr])
        if missing:
            raise RuntimeError(
                f"Outer fold {outer_fold}: test classes absent from train: {sorted(missing)}"
            )

        selected, search_table = select_distillation_hyperparameters(
            Xp.iloc[tr].reset_index(drop=True),
            Xt.iloc[tr].reset_index(drop=True),
            y[tr],
            groups[tr],
            classes,
            candidates=candidates,
            inner_folds=args.inner_folds,
            seed=args.seed + outer_fold * 100000,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
        )

        selected_alpha, selected_temp = selected

        y_tr_idx = np.asarray(
            [class_to_idx[label] for label in y[tr]],
            dtype=np.int64,
        )

        baseline_prep, baseline_model = train_student(
            Xp.iloc[tr].reset_index(drop=True),
            y_tr_idx,
            len(classes),
            teacher_probs=None,
            alpha=1.0,
            temperature=1.0,
            seed=args.seed + outer_fold * 1000,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
        )

        oof_baseline[te] = predict_student(
            baseline_prep,
            baseline_model,
            Xp.iloc[te],
        )

        teacher_soft_train = None
        if selected_alpha < 1.0:
            teacher_soft_train = crossfit_teacher_probs(
                Xt.iloc[tr].reset_index(drop=True),
                y[tr],
                groups[tr],
                classes,
                n_splits=args.inner_folds,
                seed=args.seed + outer_fold * 10000,
            )

        distilled_prep, distilled_model = train_student(
            Xp.iloc[tr].reset_index(drop=True),
            y_tr_idx,
            len(classes),
            teacher_probs=teacher_soft_train,
            alpha=selected_alpha,
            temperature=selected_temp,
            seed=args.seed + outer_fold * 1000,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            dropout=args.dropout,
        )

        oof_distilled[te] = predict_student(
            distilled_prep,
            distilled_model,
            Xp.iloc[te],
        )

        teacher_imp = SimpleImputer(strategy="median")
        Xt_tr = teacher_imp.fit_transform(Xt.iloc[tr])
        Xt_te = teacher_imp.transform(Xt.iloc[te])

        teacher = rf_teacher(args.seed + outer_fold * 1000 + 99)
        teacher.fit(Xt_tr, y[tr])

        oof_teacher[te] = align_teacher_proba(
            teacher,
            Xt_te,
            classes,
        )

        fold_id[te] = outer_fold

        baseline_fold = probability_metrics(
            y[te],
            oof_baseline[te],
            classes,
        )
        distilled_fold = probability_metrics(
            y[te],
            oof_distilled[te],
            classes,
        )

        fold_records.append({
            "fold": outer_fold,
            "selected_alpha": selected_alpha,
            "selected_temperature": selected_temp,
            "baseline_balanced_accuracy": baseline_fold["balanced_accuracy"],
            "distilled_balanced_accuracy": distilled_fold["balanced_accuracy"],
            "baseline_macro_f1": baseline_fold["macro_f1"],
            "distilled_macro_f1": distilled_fold["macro_f1"],
            "train_experiments": sorted(train_groups),
            "test_experiments": sorted(test_groups),
            "hyperparameter_search": search_table,
        })

        print(json.dumps({
            "completed_outer_fold": outer_fold,
            "selected_alpha": selected_alpha,
            "selected_temperature": selected_temp,
            "baseline_balanced_accuracy": baseline_fold["balanced_accuracy"],
            "distilled_balanced_accuracy": distilled_fold["balanced_accuracy"],
        }, indent=2), flush=True)

    trace_baseline = probability_metrics(y, oof_baseline, classes)
    trace_distilled = probability_metrics(y, oof_distilled, classes)
    trace_teacher = probability_metrics(y, oof_teacher, classes)

    y_exp, p_base_exp, _ = fuse_by_experiment(
        df[["experiment_id"]],
        y,
        oof_baseline,
    )
    _, p_dist_exp, _ = fuse_by_experiment(
        df[["experiment_id"]],
        y,
        oof_distilled,
    )
    _, p_teacher_exp, _ = fuse_by_experiment(
        df[["experiment_id"]],
        y,
        oof_teacher,
    )

    exp_baseline = probability_metrics(y_exp, p_base_exp, classes)
    exp_distilled = probability_metrics(y_exp, p_dist_exp, classes)
    exp_teacher = probability_metrics(y_exp, p_teacher_exp, classes)

    selected_counts = {}
    for record in fold_records:
        key = (
            f"alpha={record['selected_alpha']},"
            f"T={record['selected_temperature']}"
        )
        selected_counts[key] = selected_counts.get(key, 0) + 1

    fold_base_ba = [r["baseline_balanced_accuracy"] for r in fold_records]
    fold_dist_ba = [r["distilled_balanced_accuracy"] for r in fold_records]
    fold_base_f1 = [r["baseline_macro_f1"] for r in fold_records]
    fold_dist_f1 = [r["distilled_macro_f1"] for r in fold_records]

    results = {
        "target": target,
        "teacher_profile": args.teacher_profile,
        "classes": classes,
        "sample_count": int(len(df)),
        "experiment_count": int(df["experiment_id"].nunique()),
        "teacher_predictor_count": int(len(teacher_cols)),
        "proxy_predictor_count": int(len(proxy_cols)),
        "eligible_class_experiment_counts": {
            str(k): int(v)
            for k, v in exp_per_class.items()
            if k in eligible
        },
        "excluded_ineligible_classes": {
            str(k): int(v)
            for k, v in excluded.items()
        },
        "hyperparameter_candidates": [
            {"alpha": a, "temperature": t}
            for a, t in candidates
        ],
        "selected_hyperparameter_counts": selected_counts,
        "client_trace_level": {
            "baseline_proxy_student": trace_baseline,
            "distilled_proxy_student": trace_distilled,
            "endpoint_teacher_upper_bound": trace_teacher,
        },
        "experiment_fused_level": {
            "baseline_proxy_student": exp_baseline,
            "distilled_proxy_student": exp_distilled,
            "endpoint_teacher_upper_bound": exp_teacher,
        },
        "fold_stability": {
            "baseline_balanced_accuracy_mean": float(np.mean(fold_base_ba)),
            "baseline_balanced_accuracy_std": float(np.std(fold_base_ba, ddof=0)),
            "distilled_balanced_accuracy_mean": float(np.mean(fold_dist_ba)),
            "distilled_balanced_accuracy_std": float(np.std(fold_dist_ba, ddof=0)),
            "baseline_macro_f1_mean": float(np.mean(fold_base_f1)),
            "baseline_macro_f1_std": float(np.std(fold_base_f1, ddof=0)),
            "distilled_macro_f1_mean": float(np.mean(fold_dist_f1)),
            "distilled_macro_f1_std": float(np.std(fold_dist_f1, ddof=0)),
        },
        "folds": fold_records,
        "guardrails": {
            "outer_split_unit": "experiment_id",
            "inner_split_unit": "experiment_id",
            "student_outer_test_input": "proxy_only",
            "endpoint_features_in_student_outer_test": False,
            "teacher_soft_targets_for_student_training": "experiment-disjoint cross-fitted",
            "hyperparameter_selection": "inner experiment-disjoint CV only",
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"endpoint_assisted_nested_"
        f"{target}_"
        f"{args.teacher_profile}"
    )

    result_path = out_dir / f"{stem}_results.json"
    result_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    pred = df[["experiment_id", "client_id", target]].copy()
    pred["fold"] = fold_id

    for j, c in enumerate(classes):
        pred[f"baseline_p__{c}"] = oof_baseline[:, j]
        pred[f"distilled_p__{c}"] = oof_distilled[:, j]
        pred[f"teacher_p__{c}"] = oof_teacher[:, j]

    pred_path = out_dir / f"{stem}_oof_predictions.csv"
    pred.to_csv(pred_path, index=False)

    print("\nFINAL RESULT\n", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print(f"\nSaved: {result_path}", flush=True)
    print(f"Saved: {pred_path}", flush=True)


if __name__ == "__main__":
    main()
