from ai_fingerprint import registry


def test_native_pytorch_cnn_contains_resnet_architecture():
    assert "resnet" in registry.architectures_for(
        "pytorch",
        "cnn",
        "native",
    )
    assert "resnet18" in registry.variants_for(
        "pytorch",
        "cnn",
        "resnet",
        "native",
    )


def test_hierarchy_for_resnet50():
    assert registry.hierarchy_for_variant("resnet50") == {
        "family": "cnn",
        "architecture": "resnet",
        "variant": "resnet50",
    }


def test_artifact_runtime_exposes_broader_families():
    families = registry.families_for_framework(
        "pytorch",
        "onnxruntime",
    )
    for family in {
        "cnn",
        "rnn",
        "transformer",
        "autoencoder",
        "gnn",
        "diffusion",
        "gan",
        "mlp",
        "state_space",
    }:
        assert family in families


def test_native_runtime_only_lists_native_implementations():
    families = registry.families_for_framework(
        "pytorch",
        "native",
    )
    assert "cnn" in families
    assert "rnn" in families
    assert "transformer" in families
    assert "gnn" not in families


def test_application_specific_dataset_mapping():
    assert "uci_har" in registry.datasets_for(
        "lstm",
        "activity_recognition",
        "lstm_2layer",
    )
    assert "imdb" not in registry.datasets_for(
        "lstm",
        "activity_recognition",
        "lstm_2layer",
    )
    assert "imdb" in registry.datasets_for(
        "lstm",
        "text_classification",
        "lstm_2layer",
    )


def test_gnn_synthetic_graph_mapping():
    assert "synthetic_graph" in registry.datasets_for(
        "gcn",
        "node_classification",
        "gcn_2layer",
    )


def test_legacy_model_label_upgrade():
    assert registry.upgrade_legacy_model_labels("resnet18") == (
        "resnet",
        "resnet18",
    )
    assert registry.upgrade_legacy_model_labels("lstm") == (
        "lstm",
        "lstm_2layer",
    )


def test_variational_autoencoders_support_reconstruction_and_anomaly_detection():
    for variant in ("vae_fc", "vae_conv", "beta_vae"):
        assert registry.applications_for("variational_autoencoder", variant) == [
            "anomaly_detection",
            "reconstruction",
        ]
        for application in ("reconstruction", "anomaly_detection"):
            datasets = registry.datasets_for(
                "variational_autoencoder", application, variant
            )
            assert "fashion_mnist" in datasets
            assert "cifar10" in datasets
