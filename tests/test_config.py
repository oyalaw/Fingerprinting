import copy

import pytest

from ai_fingerprint.config import DEFAULT_CONFIG, ConfigError, validate_config


def test_default_config_is_valid():
    config = copy.deepcopy(DEFAULT_CONFIG)
    validate_config(config)


def test_manual_dataset_requires_local_path():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ai"].update(
        {
            "dataset": "imagenet",
            "application": "image_classification",
        }
    )
    with pytest.raises(ConfigError):
        validate_config(config)



def test_invalid_variant_is_rejected():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ai"]["variant"] = "bert_base"
    with pytest.raises(ConfigError):
        validate_config(config)



def test_federated_inference_is_rejected():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["execution"]["deployment"] = "federated"
    with pytest.raises(ConfigError):
        validate_config(config)


def test_local_training_client_is_valid():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["node"]["role"] = "client"
    config["execution"]["task"] = "training"
    config["execution"]["deployment"] = "local"
    config["data"]["split"] = "train"
    validate_config(config)


def test_training_artifact_runtime_is_rejected():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["node"]["role"] = "client"
    config["execution"]["task"] = "training"
    config["execution"]["deployment"] = "local"
    config["ai"]["runtime"] = "onnxruntime"
    config["ai"]["model_artifact"] = "model.onnx"
    with pytest.raises(ConfigError):
        validate_config(config)
