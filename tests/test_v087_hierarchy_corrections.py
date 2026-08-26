from __future__ import annotations

import copy

import numpy as np
import pytest

from ai_fingerprint import registry
from ai_fingerprint.architecture_models import fisher_score_ranking
from ai_fingerprint.config import DEFAULT_CONFIG
from ai_fingerprint.proxy import DEFAULT_PROXY_CONFIG, ProxyError, validate_proxy_config


def test_transformer_native_hierarchy_has_real_branching():
    architectures = registry.architectures_for(
        "pytorch", "transformer", "native"
    )
    assert {"tiny_transformer", "bert", "distilbert", "vit"}.issubset(
        architectures
    )
    assert registry.variants_for(
        "pytorch", "transformer", "tiny_transformer", "native"
    ) == [
        "tiny_transformer_2layer",
        "tiny_transformer_4layer",
        "tiny_transformer_6layer",
    ]


def test_autoencoder_native_hierarchy_has_architecture_and_variant_branching():
    architectures = registry.architectures_for(
        "pytorch", "autoencoder", "native"
    )
    assert architectures == [
        "convolutional_autoencoder",
        "dense_autoencoder",
        "variational_autoencoder",
    ]
    assert len(
        registry.variants_for(
            "pytorch",
            "autoencoder",
            "convolutional_autoencoder",
            "native",
        )
    ) == 3
    assert len(
        registry.variants_for(
            "pytorch", "autoencoder", "dense_autoencoder", "native"
        )
    ) == 3
    assert len(
        registry.variants_for(
            "pytorch",
            "autoencoder",
            "variational_autoencoder",
            "native",
        )
    ) == 3


def test_mlp_is_native_and_depth_variants_are_exposed():
    assert "mlp" in registry.families_for_framework("pytorch", "native")
    assert registry.variants_for(
        "pytorch", "mlp", "feedforward_mlp", "native"
    ) == ["mlp_2layer", "mlp_4layer", "mlp_8layer"]


def test_legacy_variant_upgrade_is_comprehensive_for_previous_experiments():
    assert registry.upgrade_legacy_model_labels("resnet101") == (
        "resnet",
        "resnet101",
    )
    assert registry.upgrade_legacy_model_labels("mobilenet_v3_large") == (
        "mobilenet",
        "mobilenet_v3_large",
    )
    assert registry.upgrade_legacy_model_labels("efficientnet_b2") == (
        "efficientnet",
        "efficientnet_b2",
    )


def test_dell_device_labels_are_first_class_options():
    assert "dell_desktop" in registry.DEVICES
    assert "dell_laptop" in registry.DEVICES


def test_proxy_realtime_rejects_accidental_single_scale():
    config = copy.deepcopy(DEFAULT_PROXY_CONFIG)
    config["experiment"]["experiment_id"] = "TEST"
    config["capture"]["interface"] = "lo"
    config["capture"]["client_ips"] = ["127.0.0.2"]
    config["capture"]["window_sizes_sec"] = [5.0]
    with pytest.raises(ProxyError, match="multiscale"):
        validate_proxy_config(config)

    config["capture"]["allow_single_scale"] = True
    validate_proxy_config(config)


def test_fisher_score_ranks_discriminating_feature_first():
    X = np.asarray(
        [
            [0.0, 10.0, 5.0],
            [0.1, 11.0, 5.1],
            [9.9, 10.5, 4.9],
            [10.0, 11.5, 5.0],
        ],
        dtype=np.float64,
    )
    labels = ["a", "a", "b", "b"]
    ranking = fisher_score_ranking(X, labels, ["signal", "noise", "flat"])
    assert ranking[0]["feature"] == "signal"
    assert ranking[0]["score"] > ranking[1]["score"]


def _base_training_config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["node"]["role"] = "client"
    config["execution"]["task"] = "training"
    config["execution"]["deployment"] = "local"
    config["ai"]["framework"] = "pytorch"
    config["ai"]["runtime"] = "native"
    config["execution"]["batch_size"] = 1
    return config


def test_new_pytorch_tiny_transformer_variant_executes_one_batch():
    pytest.importorskip("torch")
    from ai_fingerprint.workloads.pytorch_backend import PyTorchWorkload

    config = _base_training_config()
    config["ai"].update(
        {
            "family": "transformer",
            "architecture": "tiny_transformer",
            "variant": "tiny_transformer_4layer",
            "application": "text_classification",
            "dataset": "synthetic_text",
            "num_classes": 2,
            "vocab_size": 100,
            "max_text_length": 8,
        }
    )
    workload = PyTorchWorkload(config)
    x = np.arange(8, dtype=np.int64).reshape(1, 8) % 100
    y = np.asarray([1], dtype=np.int64)
    metrics = workload.train_batch(x, y)
    assert np.isfinite(metrics["loss"])


def test_new_pytorch_autoencoder_and_mlp_execute():
    pytest.importorskip("torch")
    from ai_fingerprint.workloads.pytorch_backend import PyTorchWorkload

    ae = _base_training_config()
    ae["ai"].update(
        {
            "family": "autoencoder",
            "architecture": "convolutional_autoencoder",
            "variant": "convolutional_autoencoder_6layer",
            "application": "reconstruction",
            "dataset": "synthetic_image",
            "input_size": 8,
        }
    )
    workload = PyTorchWorkload(ae)
    x = np.random.default_rng(1).random((1, 3, 8, 8), dtype=np.float32)
    assert np.isfinite(workload.train_batch(x, x)["loss"])

    mlp = _base_training_config()
    mlp["ai"].update(
        {
            "family": "mlp",
            "architecture": "feedforward_mlp",
            "variant": "mlp_4layer",
            "application": "image_classification",
            "dataset": "synthetic_image",
            "input_size": 8,
            "num_classes": 10,
        }
    )
    workload = PyTorchWorkload(mlp)
    y = np.asarray([2], dtype=np.int64)
    assert np.isfinite(workload.train_batch(x, y)["loss"])
