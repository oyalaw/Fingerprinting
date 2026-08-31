from __future__ import annotations

import copy
import csv

import numpy as np
import pytest

from ai_fingerprint import registry
from ai_fingerprint.architecture_models import (
    load_bundle,
    predict_hierarchy,
    train_hierarchy_bundle,
)
from ai_fingerprint.config import DEFAULT_CONFIG, validate_config
from ai_fingerprint.dataset_manager import DatasetManager


def _mlm_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["node"]["role"] = "client"
    config["execution"].update({
        "task": "training",
        "deployment": "local",
        "batch_size": 2,
        "repetitions": 10,
        "warmup": 0,
        "seed": 123,
    })
    config["data"].update({"split": "train", "shuffle": False})
    config["ai"].update({
        "framework": "pytorch",
        "runtime": "native",
        "family": "transformer",
        "architecture": "tiny_transformer",
        "variant": "tiny_transformer_2layer",
        "application": "masked_language_modeling",
        "dataset": "synthetic_text",
        "num_classes": 2,
        "vocab_size": 64,
        "max_text_length": 12,
    })
    config["masked_language_modeling"] = {"mask_probability": 0.25}
    return config


def test_transformer_variants_expose_real_second_application():
    for architecture, variant in (
        ("tiny_transformer", "tiny_transformer_2layer"),
        ("bert", "bert_base"),
        ("distilbert", "distilbert_base"),
    ):
        assert registry.applications_for(architecture, variant) == [
            "masked_language_modeling",
            "text_classification",
        ]

    assert registry.applications_for("vit", "vit_b16") == [
        "image_classification"
    ]


def test_mlm_config_and_dataset_generate_masked_token_targets():
    config = _mlm_config()
    validate_config(config)
    manager = DatasetManager(config)
    inputs, targets = manager.sample_training_batch()

    assert inputs.shape == (2, 12)
    assert targets.shape == (2, 12)
    masked = targets != -100
    assert np.all(masked.sum(axis=1) >= 1)
    assert np.all(inputs[masked] == 1)
    assert np.all(targets[masked] >= 2)


def test_pytorch_tiny_transformer_mlm_executes_one_batch():
    pytest.importorskip("torch")
    from ai_fingerprint.workloads.pytorch_backend import PyTorchWorkload

    config = _mlm_config()
    manager = DatasetManager(config)
    x, y = manager.sample_training_batch()
    workload = PyTorchWorkload(config)
    metrics = workload.train_batch(x, y)

    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["evaluated_tokens"] == int(np.sum(y != -100))

    evaluated = workload.evaluate_batch(x, y)
    assert np.isfinite(evaluated["loss"])
    assert evaluated["evaluated_tokens"] == int(np.sum(y != -100))


def _write_application_xy(tmp_path):
    x_path = tmp_path / "fingerprinting_X_proxy.csv"
    y_path = tmp_path / "fingerprinting_Y_ground_truth.csv"
    x_fields = [
        "row_id",
        "experiment_id",
        "row_type",
        "bytes_per_second",
        "packet_size_mean",
        "iat_sec_mean",
    ]
    y_fields = [
        "row_id",
        "experiment_id",
        "family",
        "architecture",
        "variant",
        "application",
    ]
    x_rows = []
    y_rows = []
    for application, base in (
        ("text_classification", 1000.0),
        ("masked_language_modeling", 9000.0),
    ):
        for exp in range(4):
            row_id = f"{application}-{exp}"
            experiment_id = f"{application}-exp{exp}"
            x_rows.append({
                "row_id": row_id,
                "experiment_id": experiment_id,
                "row_type": "overall",
                "bytes_per_second": base + exp * 10,
                "packet_size_mean": 500 + (100 if application == "masked_language_modeling" else 0),
                "iat_sec_mean": 0.003 if application == "masked_language_modeling" else 0.001,
            })
            y_rows.append({
                "row_id": row_id,
                "experiment_id": experiment_id,
                "family": "transformer",
                "architecture": "tiny_transformer",
                "variant": "tiny_transformer_2layer",
                "application": application,
            })

    with x_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=x_fields)
        writer.writeheader()
        writer.writerows(x_rows)
    with y_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=y_fields)
        writer.writeheader()
        writer.writerows(y_rows)
    return x_path, y_path


def test_hierarchy_trains_and_predicts_application_stage(tmp_path):
    x_path, y_path = _write_application_xy(tmp_path)
    result = train_hierarchy_bundle(
        x_path,
        y_path,
        tmp_path / "models",
        mode="final",
        feature_mode="full",
    )
    bundle = load_bundle(result["bundle_path"])
    key = "transformer::tiny_transformer::tiny_transformer_2layer"
    stage = bundle["stages"]["application_by_parent"][key]
    assert stage["kind"] == "classifier"
    assert stage["evaluation"]["status"] == "evaluated"

    prediction = predict_hierarchy(bundle, {
        "bytes_per_second": 9050.0,
        "packet_size_mean": 600.0,
        "iat_sec_mean": 0.003,
    })
    assert prediction["family"]["label"] == "transformer"
    assert prediction["architecture"]["label"] == "tiny_transformer"
    assert prediction["variant"]["label"] == "tiny_transformer_2layer"
    assert prediction["application"]["label"] == "masked_language_modeling"
