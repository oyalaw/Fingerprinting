import copy

from ai_fingerprint.config import DEFAULT_CONFIG
from ai_fingerprint.dataset_manager import DatasetManager


def test_synthetic_image_shape():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ai"]["dataset"] = "synthetic_image"
    config["ai"]["application"] = "image_classification"
    config["ai"]["input_size"] = 32
    config["execution"]["batch_size"] = 2

    manager = DatasetManager(config)
    sample = manager.sample()

    assert sample.shape == (2, 3, 32, 32)


def test_synthetic_sequence_shape():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["ai"].update(
        {
            "family": "rnn",
            "architecture": "lstm",
            "variant": "lstm_2layer",
            "application": "activity_recognition",
            "dataset": "synthetic_sequence",
            "sequence_length": 20,
            "input_dim": 6,
        }
    )
    config["execution"]["batch_size"] = 3

    manager = DatasetManager(config)
    sample = manager.sample()

    assert sample.shape == (3, 20, 6)



def test_synthetic_training_batch_has_targets():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["node"]["role"] = "client"
    config["execution"]["task"] = "training"
    config["execution"]["deployment"] = "local"
    config["data"]["split"] = "train"
    config["ai"]["input_size"] = 16
    config["execution"]["batch_size"] = 4

    manager = DatasetManager(config)
    inputs, targets = manager.sample_training_batch()

    assert inputs.shape == (4, 3, 16, 16)
    assert targets.shape == (4,)
    assert targets.dtype.kind in {"i", "u"}
