from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


POLICY_VERSION = "1.3"


class FederatedPolicyError(ValueError):
    pass


def build_training_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the server-authoritative FL training policy.

    These values are intentionally limited to workload/training-shape controls
    that must be coordinated across clients for reproducible experiments. The
    proxy remains label-blind and does not parse application payloads.
    """
    ai = config.get("ai", {})
    execution = config.get("execution", {})
    federated = config.get("federated", {})
    experiment_id = str(config.get("experiment", {}).get("experiment_id", ""))

    partition = config.get("data", {}).get("partition", {}) or {}
    anomaly = config.get("anomaly_detection", {}) or {}
    rounds = int(federated.get("rounds", 100))
    requested_mode = str(federated.get("mode", "full_scale")).strip().lower()
    # Backward compatibility for programmatic smoke tests that historically
    # changed only rounds. The normal runner calls validate_config first, where
    # an explicitly invalid full_scale value is still rejected.
    effective_mode = "smoke_test" if rounds < 100 else requested_mode
    core = {
        "policy_version": POLICY_VERSION,
        "experiment_id": experiment_id,
        "input_size": int(ai.get("input_size", 224)),
        "batch_size": int(execution.get("batch_size", 1)),
        "learning_rate": float(execution.get("learning_rate", 0.001)),
        "mode": effective_mode,
        "rounds": rounds,
        "local_epochs": int(federated.get("local_epochs", 1)),
        "steps_per_epoch": int(federated.get("steps_per_epoch", 10)),
        "partition_type": str(partition.get("type", "iid")).strip().lower(),
        "partition_alpha": float(partition.get("alpha", 0.5)),
        "partition_seed": int(partition.get("seed") if partition.get("seed") is not None else execution.get("seed", 42)),
        "partition_client_count": int(partition.get("client_count", federated.get("expected_clients", 1))),
        "partition_disjoint": bool(partition.get("disjoint", True)),
        "anomaly_labels": [int(value) for value in anomaly.get("anomaly_labels", [9])],
        "anomaly_calibration_fraction": float(anomaly.get("calibration_fraction", 0.10)),
        "anomaly_calibration_seed_offset": int(anomaly.get("calibration_seed_offset", 73001)),
        "anomaly_threshold_percentile": float(anomaly.get("threshold_percentile", 95.0)),
        "anomaly_calibration_batches": int(anomaly.get("calibration_batches", 10)),
        "anomaly_evaluation_batches": int(anomaly.get("evaluation_batches", 10)),
        "anomaly_evaluation_batch_size": int(anomaly.get("evaluation_batch_size", 32)),
    }
    validate_training_policy(core, expected_experiment_id=experiment_id)
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    core["policy_id"] = hashlib.sha256(canonical).hexdigest()
    core["authority"] = "server"
    return core


def validate_training_policy(
    policy: Dict[str, Any],
    *,
    expected_experiment_id: str | None = None,
) -> None:
    if not isinstance(policy, dict):
        raise FederatedPolicyError("Federated training policy must be a mapping")
    if str(policy.get("policy_version", "")) != POLICY_VERSION:
        raise FederatedPolicyError(
            f"Unsupported federated training policy version: {policy.get('policy_version')!r}"
        )
    experiment_id = str(policy.get("experiment_id", "")).strip()
    if not experiment_id:
        raise FederatedPolicyError("Federated training policy is missing experiment_id")
    if expected_experiment_id is not None and experiment_id != str(expected_experiment_id):
        raise FederatedPolicyError(
            "Federated training policy experiment mismatch: "
            f"expected={expected_experiment_id!r}, received={experiment_id!r}"
        )

    mode = str(policy.get("mode", "full_scale")).strip().lower()
    if mode not in {"full_scale", "smoke_test"}:
        raise FederatedPolicyError("Federated training policy mode must be full_scale or smoke_test")

    for field in ("input_size", "batch_size", "rounds", "local_epochs", "steps_per_epoch"):
        try:
            value = int(policy[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise FederatedPolicyError(f"Federated training policy has invalid {field}") from exc
        if value <= 0:
            raise FederatedPolicyError(f"Federated training policy {field} must be positive")

    rounds = int(policy["rounds"])
    if mode == "full_scale" and rounds < 100:
        raise FederatedPolicyError("full_scale policy requires at least 100 rounds")
    if mode == "smoke_test" and rounds > 99:
        raise FederatedPolicyError("smoke_test policy requires 1 to 99 rounds")

    try:
        learning_rate = float(policy["learning_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FederatedPolicyError(
            "Federated training policy has invalid learning_rate"
        ) from exc
    if learning_rate <= 0:
        raise FederatedPolicyError("Federated training policy learning_rate must be positive")

    partition_type = str(policy.get("partition_type", "iid")).strip().lower()
    if partition_type not in {"iid", "non_iid"}:
        raise FederatedPolicyError("partition_type must be iid or non_iid")
    try:
        alpha = float(policy.get("partition_alpha", 0.5))
        partition_seed = int(policy.get("partition_seed", 42))
        client_count = int(policy.get("partition_client_count", 1))
    except (TypeError, ValueError) as exc:
        raise FederatedPolicyError("Invalid federated data-partition policy") from exc
    if alpha <= 0:
        raise FederatedPolicyError("partition_alpha must be positive")
    if client_count <= 0:
        raise FederatedPolicyError("partition_client_count must be positive")

    anomaly_labels = policy.get("anomaly_labels", [9])
    if not isinstance(anomaly_labels, (list, tuple)) or not anomaly_labels:
        raise FederatedPolicyError("anomaly_labels must contain at least one integer")
    try:
        [int(value) for value in anomaly_labels]
        calibration_fraction = float(policy.get("anomaly_calibration_fraction", 0.10))
        int(policy.get("anomaly_calibration_seed_offset", 73001))
        percentile = float(policy.get("anomaly_threshold_percentile", 95.0))
        calibration_batches = int(policy.get("anomaly_calibration_batches", 10))
        evaluation_batches = int(policy.get("anomaly_evaluation_batches", 10))
        evaluation_batch_size = int(policy.get("anomaly_evaluation_batch_size", 32))
    except (TypeError, ValueError) as exc:
        raise FederatedPolicyError("Invalid anomaly-detection evaluation policy") from exc
    if not (0.0 < calibration_fraction < 1.0):
        raise FederatedPolicyError("anomaly_calibration_fraction must be between 0 and 1")
    if not (0.0 < percentile < 100.0):
        raise FederatedPolicyError("anomaly_threshold_percentile must be between 0 and 100")
    if min(calibration_batches, evaluation_batches, evaluation_batch_size) <= 0:
        raise FederatedPolicyError("anomaly evaluation batch controls must be positive")

    supplied_policy_id = str(policy.get("policy_id", "")).strip()
    if supplied_policy_id:
        core = {
            key: policy[key]
            for key in (
                "policy_version",
                "experiment_id",
                "input_size",
                "batch_size",
                "mode",
                "learning_rate",
                "rounds",
                "local_epochs",
                "steps_per_epoch",
                "partition_type",
                "partition_alpha",
                "partition_seed",
                "partition_client_count",
                "partition_disjoint",
                "anomaly_labels",
                "anomaly_calibration_fraction",
                "anomaly_calibration_seed_offset",
                "anomaly_threshold_percentile",
                "anomaly_calibration_batches",
                "anomaly_evaluation_batches",
                "anomaly_evaluation_batch_size",
            )
        }
        canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_policy_id = hashlib.sha256(canonical).hexdigest()
        if supplied_policy_id != expected_policy_id:
            raise FederatedPolicyError(
                "Federated training policy digest does not match its contents"
            )


def apply_training_policy(config: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a validated server policy to a client configuration in place."""
    expected = str(config.get("experiment", {}).get("experiment_id", ""))
    validate_training_policy(policy, expected_experiment_id=expected)

    config.setdefault("ai", {})["input_size"] = int(policy["input_size"])
    execution = config.setdefault("execution", {})
    execution["batch_size"] = int(policy["batch_size"])
    execution["learning_rate"] = float(policy["learning_rate"])

    federated = config.setdefault("federated", {})
    federated["mode"] = str(policy.get("mode", "full_scale"))
    federated["rounds"] = int(policy["rounds"])
    federated["local_epochs"] = int(policy["local_epochs"])
    federated["steps_per_epoch"] = int(policy["steps_per_epoch"])
    federated["policy_source"] = "server"
    federated["policy_id"] = str(policy.get("policy_id", ""))
    federated["policy_applied"] = True
    partition = config.setdefault("data", {}).setdefault("partition", {})
    partition["type"] = str(policy.get("partition_type", "iid"))
    partition["alpha"] = float(policy.get("partition_alpha", 0.5))
    partition["seed"] = int(policy.get("partition_seed", 42))
    partition["client_count"] = int(policy.get("partition_client_count", 1))
    partition["disjoint"] = bool(policy.get("partition_disjoint", True))
    partition["source"] = "server"
    anomaly = config.setdefault("anomaly_detection", {})
    anomaly["anomaly_labels"] = [int(value) for value in policy.get("anomaly_labels", [9])]
    anomaly["calibration_fraction"] = float(policy.get("anomaly_calibration_fraction", 0.10))
    anomaly["calibration_seed_offset"] = int(policy.get("anomaly_calibration_seed_offset", 73001))
    anomaly["threshold_percentile"] = float(policy.get("anomaly_threshold_percentile", 95.0))
    anomaly["calibration_batches"] = int(policy.get("anomaly_calibration_batches", 10))
    anomaly["evaluation_batches"] = int(policy.get("anomaly_evaluation_batches", 10))
    anomaly["evaluation_batch_size"] = int(policy.get("anomaly_evaluation_batch_size", 32))
    return config


def write_received_policy(config: Dict[str, Any], policy: Dict[str, Any]) -> Path:
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "server_training_policy.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2, sort_keys=True)
    return path


def write_effective_config(config: Dict[str, Any]) -> Path:
    """Persist the post-handshake client configuration actually executed."""
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config_effective.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path
