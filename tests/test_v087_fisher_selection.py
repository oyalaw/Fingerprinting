from __future__ import annotations

import csv
from pathlib import Path

from ai_fingerprint.architecture_models import (
    _evaluate_grouped_with_fisher,
    fisher_score_ranking,
    load_bundle,
    predict_hierarchy,
    select_fisher_features,
    train_hierarchy_bundle,
)


def test_fisher_ranking_prefers_class_separating_feature():
    rows = []
    for index in range(20):
        rows.append(
            {
                "_family": "cnn" if index < 10 else "autoencoder",
                "family_signal": 10.0 if index < 10 else 100.0,
                "noise": float(index % 3),
            }
        )

    ranking = fisher_score_ranking(
        rows,
        ["family_signal", "noise"],
        "_family",
    )
    assert ranking[0]["feature"] == "family_signal"
    assert ranking[0]["fisher_score"] > ranking[1]["fisher_score"]


def test_stage_specific_fisher_selection_can_differ():
    rows = []
    for family in ("cnn", "autoencoder"):
        for architecture in (
            ("resnet", "mobilenet")
            if family == "cnn"
            else ("cae",)
        ):
            for i in range(20):
                rows.append(
                    {
                        "_family": family,
                        "_architecture": architecture,
                        "family_signal": (
                            100.0
                            if family == "autoencoder"
                            else 10.0
                        ) + (i % 2) * 0.01,
                        "architecture_signal": (
                            50.0
                            if architecture == "resnet"
                            else 5.0
                        ) + (i % 2) * 0.01,
                        "noise": float(i % 7),
                    }
                )

    family_selected, _ = select_fisher_features(
        rows,
        ["family_signal", "architecture_signal", "noise"],
        "_family",
        top_k=1,
        min_features=1,
    )
    cnn_rows = [row for row in rows if row["_family"] == "cnn"]
    arch_selected, _ = select_fisher_features(
        cnn_rows,
        ["family_signal", "architecture_signal", "noise"],
        "_architecture",
        top_k=1,
        min_features=1,
    )

    assert family_selected == ["family_signal"]
    assert arch_selected == ["architecture_signal"]


def _write_hierarchical_dataset(tmp_path: Path):
    x_path = tmp_path / "fingerprinting_X_proxy.csv"
    y_path = tmp_path / "fingerprinting_Y_ground_truth.csv"

    x_fields = [
        "row_id",
        "experiment_id",
        "client_capture_id",
        "row_type",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "window_size_sec",
        "family_signal",
        "architecture_signal",
        "variant_signal",
        "noise_a",
        "noise_b",
    ]
    y_fields = [
        "row_id",
        "experiment_id",
        "family",
        "architecture",
        "variant",
    ]

    x_rows = []
    y_rows = []
    row_id = 0

    hierarchy = [
        ("cnn", "resnet", "resnet18", 10, 100, 1000),
        ("cnn", "resnet", "resnet101", 10, 100, 2000),
        ("cnn", "mobilenet", "mobilenetv2", 10, 200, 3000),
        ("autoencoder", "cae", "cae4", 50, 400, 4000),
    ]

    for family, architecture, variant, f_sig, a_sig, v_sig in hierarchy:
        for run in range(3):
            experiment_id = f"{family}_{architecture}_{variant}_{run}"
            for client in range(2):
                row_id += 1
                jitter = run * 0.01 + client * 0.001
                x_rows.append(
                    {
                        "row_id": row_id,
                        "experiment_id": experiment_id,
                        "client_capture_id": f"trace_{client}",
                        "row_type": "overall",
                        "window_index": -1,
                        "window_start_sec": 0,
                        "window_end_sec": 10,
                        "window_size_sec": 0,
                        "family_signal": f_sig + jitter,
                        "architecture_signal": a_sig + jitter,
                        "variant_signal": v_sig + jitter,
                        "noise_a": run + client,
                        "noise_b": (run + client) % 2,
                    }
                )
                y_rows.append(
                    {
                        "row_id": row_id,
                        "experiment_id": experiment_id,
                        "family": family,
                        "architecture": architecture,
                        "variant": variant,
                    }
                )

    with x_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=x_fields)
        writer.writeheader()
        writer.writerows(x_rows)

    with y_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=y_fields)
        writer.writeheader()
        writer.writerows(y_rows)

    return x_path, y_path


def test_bundle_uses_stage_specific_feature_vectors(tmp_path):
    x_path, y_path = _write_hierarchical_dataset(tmp_path)
    result = train_hierarchy_bundle(
        x_path,
        y_path,
        tmp_path / "models",
        mode="final",
        feature_mode="full",
        fisher_top_k=1,
        fisher_min_features=1,
    )

    bundle = load_bundle(result["bundle_path"])
    family_stage = bundle["stages"]["family"]
    cnn_arch_stage = bundle["stages"]["architecture_by_family"]["cnn"]
    cnn_resnet_variant = bundle["stages"]["variant_by_parent"][
        "cnn::resnet"
    ]

    assert family_stage["feature_columns"] == ["family_signal"]
    assert cnn_arch_stage["feature_columns"] == ["architecture_signal"]
    assert cnn_resnet_variant["feature_columns"] == ["variant_signal"]

    prediction = predict_hierarchy(
        bundle,
        {
            "family_signal": 10.0,
            "architecture_signal": 100.0,
            "variant_signal": 2000.0,
            "noise_a": 0.0,
            "noise_b": 0.0,
        },
    )
    assert prediction["family"]["label"] == "cnn"
    assert prediction["architecture"]["label"] == "resnet"
    assert prediction["variant"]["label"] == "resnet101"

    assert Path(result["fisher_scores_csv"]).exists()


def test_grouped_evaluation_records_fold_local_selection():
    rows = []
    for label, signal in [("cnn", 10.0), ("autoencoder", 100.0)]:
        for run in range(2):
            for sample in range(2):
                rows.append(
                    {
                        "experiment_id": f"{label}_{run}",
                        "_family": label,
                        "signal": signal + sample * 0.01,
                        "noise": float(run + sample),
                    }
                )

    evaluation = _evaluate_grouped_with_fisher(
        rows,
        ["signal", "noise"],
        "_family",
        fisher_top_k=1,
        fisher_min_score=1e-6,
        fisher_min_features=1,
    )
    assert evaluation["status"] == "evaluated"
    selection = evaluation["feature_selection"]
    assert selection["fitted_inside_each_training_fold"] is True
    assert selection["selection_stability"]
