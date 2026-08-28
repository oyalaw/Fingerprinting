from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


POLICY_VERSION = "1.0"


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

    core = {
        "policy_version": POLICY_VERSION,
        "experiment_id": experiment_id,
        "input_size": int(ai.get("input_size", 224)),
        "batch_size": int(execution.get("batch_size", 1)),
        "learning_rate": float(execution.get("learning_rate", 0.001)),
        "rounds": int(federated.get("rounds", 10)),
        "local_epochs": int(federated.get("local_epochs", 1)),
        "steps_per_epoch": int(federated.get("steps_per_epoch", 10)),
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

    for field in ("input_size", "batch_size", "rounds", "local_epochs", "steps_per_epoch"):
        try:
            value = int(policy[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise FederatedPolicyError(f"Federated training policy has invalid {field}") from exc
        if value <= 0:
            raise FederatedPolicyError(f"Federated training policy {field} must be positive")

    try:
        learning_rate = float(policy["learning_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FederatedPolicyError(
            "Federated training policy has invalid learning_rate"
        ) from exc
    if learning_rate <= 0:
        raise FederatedPolicyError("Federated training policy learning_rate must be positive")

    supplied_policy_id = str(policy.get("policy_id", "")).strip()
    if supplied_policy_id:
        core = {
            key: policy[key]
            for key in (
                "policy_version",
                "experiment_id",
                "input_size",
                "batch_size",
                "learning_rate",
                "rounds",
                "local_epochs",
                "steps_per_epoch",
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
    federated["rounds"] = int(policy["rounds"])
    federated["local_epochs"] = int(policy["local_epochs"])
    federated["steps_per_epoch"] = int(policy["steps_per_epoch"])
    federated["policy_source"] = "server"
    federated["policy_id"] = str(policy.get("policy_id", ""))
    federated["policy_applied"] = True
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
