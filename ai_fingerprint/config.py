from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any, Dict

import yaml

from . import registry
from .dataset_catalog import DATASETS


class ConfigError(ValueError):
    pass


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "experiment_id": "auto",
        "output_dir": "experiments/results",
        "results_root": "experiments/results",
        "existing_output_policy": "error",
        "run_id": None,
    },
    "node": {
        "role": "server",
        "host": "10.42.0.195",
        "port": 8080,
    },
    "coordination": {
        # Out-of-band experiment coordination. Only a neutral run ID is
        # exposed to the label-blind proxy; no AI ground-truth labels.
        "enabled": True,
        "host": "10.42.0.195",
        "port": 8081,
    },
    "transport": {
        # Kept as tcp for backward-compatible scripted configs. The
        # interactive remote/federated workflow now offers TLS first.
        "kind": "tcp",
        "certfile": None,
        "keyfile": None,
        "cafile": None,
        "verify_peer": False,
        "server_hostname": None,
        "auto_generate_self_signed": False,
        "self_signed_common_name": "ai-fingerprint-server",
        "self_signed_valid_days": 30,
        "minimum_tls_version": "TLSv1_2",
    },
    "ai": {
        "framework": "pytorch",
        "runtime": "native",
        "family": "cnn",
        "architecture": "resnet",
        "variant": "resnet18",
        "application": "image_classification",
        "dataset": "synthetic_image",
        "model_artifact": None,
        "num_classes": 10,
        "input_size": 224,
        "sequence_length": 128,
        "input_dim": 9,
        "vocab_size": 10000,
        "max_text_length": 128,
        "graph_nodes": 32,
        "graph_features": 16,
    },
    "data": {
        "root": "datasets",
        "split": "test",
        "auto_download": True,
        "shuffle": True,
        "max_samples": None,
        "partition": {
            "type": "iid",
            "alpha": 0.5,
            "seed": None,
            "client_count": 1,
            "client_index": 0,
            "client_id": None,
            "disjoint": True,
            "source": "server",
        },
        "local_paths": {
            "imagenet": None,
            "coco2017": None,
        },
    },
    "execution": {
        "task": "inference",
        "deployment": "remote",
        "operation": "workload",
        "repetitions": 20,
        "warmup": 2,
        "interval_ms": 250,
        "batch_size": 1,
        "precision": "fp32",
        "seed": 42,
        "epochs": 1,
        "steps_per_epoch": 20,
        "learning_rate": 0.001,
    },
    "federated": {
        # full_scale enforces publication-scale runs of at least 100 global rounds.
        # smoke_test is intentionally restricted to 1..99 rounds for debugging.
        "mode": "full_scale",
        "rounds": 100,
        "local_epochs": 1,
        "steps_per_epoch": 10,
        "expected_clients": 2,
        "client_id": "client_1",
        "aggregation": "fedavg",
        "policy_source": "server",
        "policy_applied": False,
        "policy_id": None,
    },
    "device": {
        "label": "custom",
        "operating_system": "unknown",
    },
    "masked_language_modeling": {
        # Deterministic experiment RNG chooses masked token positions. Token 0
        # is padding, token 1 is reserved as [MASK], and -100 marks targets
        # excluded from the token-level loss.
        "mask_probability": 0.15,
    },
    "anomaly_detection": {
        # Reserve a deterministic normal-only subset from the training split
        # exclusively for threshold calibration. It never participates in
        # local model training and the test split is never used to tune tau.
        "calibration_fraction": 0.10,
        "calibration_seed_offset": 73001,
        "data_role": "train",
        # Class-label based anomaly protocol used only when
        # ai.application=anomaly_detection. Training excludes these labels;
        # held-out evaluation converts reconstruction error to anomaly scores.
        "anomaly_labels": [9],
        "threshold_percentile": 95.0,
        "calibration_batches": 10,
        "evaluation_batches": 10,
        "evaluation_batch_size": 32,
    },
    "checkpoint": {
        "enabled": True,
        "interval_rounds": 10,
        "retain_archives": 3,
        "minimum_free_disk_gb": 5.0,
    },
    "performance_logging": {
        # Ground-truth/system-characterization only. These metrics are never
        # admitted into proxy-side fingerprinting predictor sets.
        "enabled": True,
        # Client evaluates one held-out round probe before and after local
        # training. This adds two forward passes per FL round.
        "client_round_probe": True,
        # Server evaluates the aggregated global model on a separate split
        # after each FedAvg round. Evaluation is intentionally part of the
        # synchronous server round and its time is logged explicitly.
        "server_eval_batches": 10,
        "server_eval_split": "test",
        "server_evaluation_required": False,
    },
    "resource_monitor": {
        "enabled": True,
        "interval_ms": 500,
        "network_interface": None,
        "gpu_index": 0,
        "power_enabled": True,
        # Probe nvidia-smi once. Do not repeatedly invoke it on systems
        # where the binary exists but no usable NVIDIA GPU is present.
        "disable_unusable_nvidia_smi": True,
    },
    "result_collection": {
        # Copy-only central replication. Local experiment results remain the
        # authoritative originals even after a verified upload.
        "enabled": True,
        "collector_host": "10.42.0.195",
        "collector_port": 8090,
        "timeout_sec": 10.0,
        "retry_attempts": 3,
        "retry_delay_sec": 1.0,
        "shared_token": "",
        "max_archive_mb": 1024,
        "central_root": "collected_experiments",
    },
}


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def generate_experiment_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"EXP_{stamp}"


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}

    execution_supplied = supplied.get("execution", {})
    if (
        isinstance(execution_supplied, dict)
        and "mode" in execution_supplied
        and "task" not in execution_supplied
    ):
        legacy_mode = str(execution_supplied.pop("mode"))
        legacy_map = {
            "remote_inference": ("inference", "remote", "workload"),
            "local_inference": ("inference", "local", "workload"),
            "model_download": ("inference", "remote", "model_download"),
            "local_training": ("training", "local", "workload"),
            "remote_training": ("training", "remote", "workload"),
            "federated_training": ("training", "federated", "workload"),
        }
        if legacy_mode in legacy_map:
            task, deployment, operation = legacy_map[legacy_mode]
            execution_supplied["task"] = task
            execution_supplied["deployment"] = deployment
            execution_supplied["operation"] = operation

    ai_supplied = supplied.get("ai", {})
    if (
        isinstance(ai_supplied, dict)
        and "architecture" in ai_supplied
        and "variant" not in ai_supplied
    ):
        upgraded = registry.upgrade_legacy_model_labels(
            str(ai_supplied["architecture"])
        )
        if upgraded is not None:
            architecture, variant = upgraded
            ai_supplied["architecture"] = architecture
            ai_supplied["variant"] = variant

    config = deep_merge(DEFAULT_CONFIG, supplied)
    if config["experiment"]["experiment_id"] in {None, "", "auto"}:
        config["experiment"]["experiment_id"] = generate_experiment_id()
    validate_config(config)
    return config


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def validate_config(config: Dict[str, Any]) -> None:
    policy = str(
        config.get("experiment", {}).get(
            "existing_output_policy",
            "error",
        )
    ).strip().lower()
    if policy not in {"error", "archive"}:
        raise ConfigError(
            "experiment.existing_output_policy must be "
            "error or archive"
        )

    role = config["node"]["role"]
    if role not in {"client", "server"}:
        raise ConfigError("node.role must be client or server")

    runtime = config["ai"]["runtime"]
    if runtime not in registry.RUNTIMES:
        raise ConfigError(f"Unsupported runtime: {runtime}")

    framework = config["ai"]["framework"]
    if framework not in registry.FRAMEWORKS:
        raise ConfigError(f"Unsupported framework: {framework}")

    partition_cfg = config.get("data", {}).get("partition", {}) or {}
    partition_type = str(partition_cfg.get("type", "iid")).strip().lower().replace("-", "_")
    if partition_type == "noniid":
        partition_type = "non_iid"
    if partition_type not in {"iid", "non_iid"}:
        raise ConfigError("data.partition.type must be iid or non_iid")
    if partition_type == "non_iid":
        try:
            partition_alpha = float(partition_cfg.get("alpha", 0.5))
        except (TypeError, ValueError) as exc:
            raise ConfigError("data.partition.alpha must be numeric") from exc
        if partition_alpha <= 0:
            raise ConfigError("data.partition.alpha must be positive for non-IID")

    device_label = str(config.get("device", {}).get("label", "")).strip().lower()
    if device_label in {
        "ubuntu", "linux", "windows", "windows_10", "windows_11",
        "macos", "darwin",
    }:
        raise ConfigError(
            "device.label must describe hardware, not the operating system. "
            "For example use device.label=dell_desktop and "
            "device.operating_system=ubuntu."
        )

    family = config["ai"]["family"]
    architecture = config["ai"]["architecture"]
    variant = config["ai"].get("variant")
    application = config["ai"]["application"]
    dataset = config["ai"]["dataset"]

    if dataset not in DATASETS:
        raise ConfigError(f"Unknown dataset: {dataset}")

    valid_architectures = registry.architectures_for(
        framework,
        family,
        runtime,
    )
    if architecture not in valid_architectures:
        raise ConfigError(
            f"Architecture {architecture!r} is not valid for framework "
            f"{framework!r}, runtime {runtime!r}, family {family!r}"
        )

    if not variant:
        raise ConfigError(
            "ai.variant is required. The model hierarchy is "
            "family -> architecture -> variant."
        )

    valid_variants = registry.variants_for(
        framework,
        family,
        architecture,
        runtime,
    )
    if variant not in valid_variants:
        raise ConfigError(
            f"Variant {variant!r} is not valid for framework "
            f"{framework!r}, family {family!r}, "
            f"architecture {architecture!r}"
        )

    if application not in registry.applications_for(
        architecture,
        variant,
    ):
        raise ConfigError(
            f"Application {application!r} is not valid for "
            f"{family!r} -> {architecture!r} -> {variant!r}"
        )

    if dataset not in registry.datasets_for(
        architecture,
        application,
        variant,
    ):
        raise ConfigError(
            f"Dataset {dataset!r} is not valid for "
            f"{family!r} -> {architecture!r} -> {variant!r} "
            f"and application {application!r}"
        )

    if registry.requires_artifact(framework, runtime, variant):
        artifact = config["ai"].get("model_artifact")
        if not artifact:
            raise ConfigError(
                f"framework={framework!r}, runtime={runtime!r}, "
                f"family={family!r}, architecture={architecture!r}, "
                f"variant={variant!r} requires ai.model_artifact"
            )

    spec = DATASETS[dataset]
    if spec.acquisition == "manual":
        local_paths = config["data"].get("local_paths", {}) or {}
        if not local_paths.get(dataset):
            raise ConfigError(
                f"Dataset {dataset!r} requires data.local_paths.{dataset}. "
                f"Expected layout: {spec.manual_layout}"
            )

    execution = config["execution"]
    task = execution.get("task")
    deployment = execution.get("deployment")
    operation = execution.get("operation", "workload")

    if task not in registry.EXECUTION_TASKS:
        raise ConfigError(
            f"Unsupported execution task: {task!r}. "
            f"Expected one of {registry.EXECUTION_TASKS}"
        )

    valid_deployments = registry.deployments_for_task(task)
    if deployment not in valid_deployments:
        raise ConfigError(
            f"Deployment {deployment!r} is not valid for task "
            f"{task!r}. Expected one of {valid_deployments}"
        )

    if operation not in {"workload", "model_download"}:
        raise ConfigError(
            "execution.operation must be workload or model_download"
        )

    if operation == "model_download":
        if task != "inference" or deployment != "remote":
            raise ConfigError(
                "model_download requires task=inference and "
                "deployment=remote"
            )

    if task == "training":
        if runtime != "native":
            raise ConfigError(
                "Training currently requires ai.runtime=native. "
                "ONNX Runtime, TensorRT, and TFLite are treated as "
                "inference runtimes in this codebase."
            )
        if not registry.native_supported(framework, variant):
            raise ConfigError(
                f"Training requires a native implementation for "
                f"{framework!r}/{variant!r}"
            )
        if application not in registry.TRAINABLE_APPLICATIONS:
            raise ConfigError(
                f"Training is not implemented for application "
                f"{application!r}"
            )

    if application == "masked_language_modeling":
        probability = float(
            (config.get("masked_language_modeling", {}) or {}).get(
                "mask_probability", 0.15
            )
        )
        if not 0.0 < probability <= 1.0:
            raise ConfigError(
                "masked_language_modeling.mask_probability must be in (0, 1]"
            )

    kind = config["transport"]["kind"]
    if kind not in {"tcp", "tls"}:
        raise ConfigError("transport.kind must be tcp or tls")

    if kind == "tls" and role == "server":
        has_material = bool(
            config["transport"].get("certfile")
            and config["transport"].get("keyfile")
        )
        auto_generate = bool(
            config["transport"].get(
                "auto_generate_self_signed", False
            )
        )
        if not has_material and not auto_generate:
            raise ConfigError(
                "TLS server requires transport.certfile/keyfile or "
                "transport.auto_generate_self_signed=true"
            )

    if kind == "tls":
        minimum = str(
            config["transport"].get(
                "minimum_tls_version", "TLSv1_2"
            )
        )
        if minimum not in {"TLSv1_2", "TLSv1_3"}:
            raise ConfigError(
                "transport.minimum_tls_version must be TLSv1_2 or TLSv1_3"
            )

    if int(config["node"]["port"]) <= 0:
        raise ConfigError("node.port must be positive")

    if int(config["execution"]["repetitions"]) <= 0:
        raise ConfigError("execution.repetitions must be positive")

    if int(config["execution"]["batch_size"]) <= 0:
        raise ConfigError("execution.batch_size must be positive")

    if int(config["execution"].get("epochs", 1)) <= 0:
        raise ConfigError("execution.epochs must be positive")

    if int(config["execution"].get("steps_per_epoch", 1)) <= 0:
        raise ConfigError("execution.steps_per_epoch must be positive")

    if float(config["execution"].get("learning_rate", 0.001)) <= 0:
        raise ConfigError("execution.learning_rate must be positive")

    if deployment == "local" and role != "client":
        raise ConfigError(
            "Local execution uses node.role=client because no server "
            "process is required."
        )

    if deployment == "federated":
        federated = config.get("federated", {})
        rounds = int(federated.get("rounds", 0))
        mode = str(federated.get("mode", "full_scale")).strip().lower()
        if mode not in {"full_scale", "smoke_test"}:
            raise ConfigError("federated.mode must be full_scale or smoke_test")
        if rounds <= 0:
            raise ConfigError("federated.rounds must be positive")
        if mode == "full_scale" and rounds < 100:
            raise ConfigError("full_scale federated experiments require at least 100 rounds")
        if mode == "smoke_test" and rounds > 99:
            raise ConfigError("smoke_test federated experiments must use 1 to 99 rounds")
        if int(federated.get("local_epochs", 0)) <= 0:
            raise ConfigError("federated.local_epochs must be positive")
        if int(federated.get("steps_per_epoch", 0)) <= 0:
            raise ConfigError(
                "federated.steps_per_epoch must be positive"
            )
        if int(federated.get("expected_clients", 0)) <= 0:
            raise ConfigError(
                "federated.expected_clients must be positive"
            )
        if federated.get("aggregation", "fedavg") != "fedavg":
            raise ConfigError(
                "v0.6 currently implements federated.aggregation=fedavg"
            )

    anomaly_cfg = config.get("anomaly_detection", {}) or {}
    if application == "anomaly_detection":
        labels_raw = anomaly_cfg.get("anomaly_labels", [])
        if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
            raise ConfigError(
                "anomaly_detection.anomaly_labels must contain at least one class label"
            )
        try:
            anomaly_labels = sorted({int(value) for value in labels_raw})
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "anomaly_detection.anomaly_labels must contain integer labels"
            ) from exc
        num_classes = max(int(config.get("ai", {}).get("num_classes", 2)), 1)
        invalid = [value for value in anomaly_labels if value < 0 or value >= num_classes]
        if invalid:
            raise ConfigError(
                "anomaly_detection.anomaly_labels contains labels outside "
                f"[0,{num_classes - 1}]: {invalid}"
            )
        if len(anomaly_labels) >= num_classes:
            raise ConfigError(
                "anomaly_detection must leave at least one normal class for training"
            )
        calibration_fraction = float(anomaly_cfg.get("calibration_fraction", 0.10))
        if not 0.0 < calibration_fraction < 1.0:
            raise ConfigError(
                "anomaly_detection.calibration_fraction must be between 0 and 1"
            )
        data_role = str(anomaly_cfg.get("data_role", "train")).strip().lower()
        if data_role not in {"train", "calibration"}:
            raise ConfigError(
                "anomaly_detection.data_role must be train or calibration"
            )
        percentile = float(anomaly_cfg.get("threshold_percentile", 95.0))
        if not (0.0 < percentile < 100.0):
            raise ConfigError(
                "anomaly_detection.threshold_percentile must be between 0 and 100"
            )
        for field in ("calibration_batches", "evaluation_batches", "evaluation_batch_size"):
            if int(anomaly_cfg.get(field, 0)) <= 0:
                raise ConfigError(f"anomaly_detection.{field} must be positive")

    performance = config.get("performance_logging", {})
    if int(performance.get("server_eval_batches", 10)) <= 0:
        raise ConfigError(
            "performance_logging.server_eval_batches must be positive"
        )
    if str(performance.get("server_eval_split", "test")) not in {
        "train", "test", "validation"
    }:
        raise ConfigError(
            "performance_logging.server_eval_split must be train, test, or validation"
        )

    monitor = config.get("resource_monitor", {})
    if int(monitor.get("interval_ms", 500)) < 100:
        raise ConfigError(
            "resource_monitor.interval_ms must be at least 100"
        )

    if int(monitor.get("gpu_index", 0)) < 0:
        raise ConfigError("resource_monitor.gpu_index must be nonnegative")

    collection = config.get("result_collection", {}) or {}
    if bool(collection.get("enabled", True)):
        host = str(collection.get("collector_host", "")).strip()
        if not host:
            raise ConfigError("result_collection.collector_host is required when enabled")
        port = int(collection.get("collector_port", 8090))
        if not 1 <= port <= 65535:
            raise ConfigError("result_collection.collector_port must be between 1 and 65535")
        if float(collection.get("timeout_sec", 10.0)) <= 0:
            raise ConfigError("result_collection.timeout_sec must be positive")
        if int(collection.get("retry_attempts", 3)) <= 0:
            raise ConfigError("result_collection.retry_attempts must be positive")
        if float(collection.get("retry_delay_sec", 1.0)) < 0:
            raise ConfigError("result_collection.retry_delay_sec must be nonnegative")
        if float(collection.get("max_archive_mb", 1024)) <= 0:
            raise ConfigError("result_collection.max_archive_mb must be positive")
