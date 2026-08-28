from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ai_fingerprint.federated import SynchronousFedAvgCoordinator
from ai_fingerprint.fingerprinting_dataset import is_forbidden_predictor_field
from ai_fingerprint.training_metrics import (
    PerformanceLogWriter,
    aggregate_batch_metrics,
    classification_metrics,
    parameter_delta_l2_norm,
    parameter_l2_norm,
)
from ai_fingerprint.workloads.base import Workload


class DummyWorkload(Workload):
    def __init__(self):
        super().__init__({"ai": {"application": "image_classification"}})
        self.parameters = [np.asarray([0.0, 0.0], dtype=np.float32)]

    def infer(self, array: np.ndarray) -> np.ndarray:
        return np.asarray([[0.0, 1.0]], dtype=np.float32)

    def get_parameters(self) -> list[np.ndarray]:
        return [value.copy() for value in self.parameters]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.parameters = [np.asarray(value).copy() for value in parameters]


def test_parameter_norms_are_model_wide():
    values = [np.asarray([3.0, 4.0]), np.asarray([12.0])]
    assert parameter_l2_norm(values) == 13.0
    assert parameter_delta_l2_norm(values, [np.zeros(2), np.zeros(1)]) == 13.0


def test_classification_metrics_macro():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == 0.75
    assert round(metrics["precision"], 6) == round((1.0 + 2 / 3) / 2, 6)
    assert round(metrics["recall"], 6) == 0.75
    assert metrics["f1"] > 0.7


def test_aggregate_batch_metrics_uses_all_predictions():
    rows = [
        {"loss": 1.0, "_targets": [0, 1], "_predictions": [0, 1], "samples": 2},
        {"loss": 0.5, "_targets": [0, 1], "_predictions": [1, 1], "samples": 2},
    ]
    metrics = aggregate_batch_metrics(rows)
    assert metrics["loss"] == 0.75
    assert metrics["accuracy"] == 0.75
    assert metrics["evaluated_samples"] == 4


def test_round_aggregation_callback_receives_before_and_after_models():
    workload = DummyWorkload()
    calls = []

    def callback(round_index, updates, previous_global, new_global, aggregation_ms):
        calls.append((round_index, updates, previous_global, new_global, aggregation_ms))

    coordinator = SynchronousFedAvgCoordinator(
        workload=workload,
        rounds=1,
        expected_clients=1,
        on_round_aggregated=callback,
    )
    next_round, done = coordinator.submit_update(
        round_index=0,
        client_id="client_1",
        parameters=[np.asarray([2.0, 4.0], dtype=np.float32)],
        num_examples=2,
        metrics={"update_norm_l2": 1.0},
    )
    assert next_round == 1
    assert done is True
    assert len(calls) == 1
    round_index, updates, previous_global, new_global, aggregation_ms = calls[0]
    assert round_index == 0
    assert set(updates) == {"client_1"}
    assert np.allclose(previous_global[0], [0.0, 0.0])
    assert np.allclose(new_global[0], [2.0, 4.0])
    assert aggregation_ms >= 0.0


def test_performance_writer_creates_role_csvs(tmp_path: Path):
    config = {
        "experiment": {"output_dir": str(tmp_path), "experiment_id": "exp1"},
    }
    writer = PerformanceLogWriter(config)
    writer.write_client_round({"experiment_id": "exp1", "round": 0, "client_id": "client_1"})
    path = tmp_path / "round_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["experiment_id"] == "exp1"
    assert rows[0]["client_id"] == "client_1"


def test_training_performance_fields_are_forbidden_proxy_predictors():
    for field in [
        "precision",
        "recall",
        "f1",
        "mse",
        "mae",
        "kl_loss",
        "update_norm_l2",
        "global_model_norm_l2",
        "aggregation_time_ms",
    ]:
        assert is_forbidden_predictor_field(field)


def test_reconstruction_dataset_uses_input_as_target():
    from ai_fingerprint.dataset_manager import DatasetManager

    class Source:
        def get_with_target(self, index):
            image = np.ones((3, 4, 4), dtype=np.float32) * index
            return image, np.asarray(7, dtype=np.int64)

    manager = object.__new__(DatasetManager)
    manager.config = {
        "ai": {"application": "reconstruction"},
        "execution": {"batch_size": 2},
    }
    manager.source = Source()
    indices = iter([1, 2])
    manager._next_index = lambda: next(indices)

    inputs, targets = manager.sample_training_batch()
    assert inputs.shape == targets.shape == (2, 3, 4, 4)
    assert np.allclose(inputs, targets)
