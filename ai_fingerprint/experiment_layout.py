from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


EXP_RE = re.compile(r"^exp(?P<number>[1-9][0-9]*)$", re.IGNORECASE)


class ExperimentLayoutError(ValueError):
    pass


def safe_component(value: Any, field: str = "component") -> str:
    text = str(value or "").strip().lower()
    if not text:
        raise ExperimentLayoutError(f"{field} must not be empty")
    safe = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in text
    ).strip("._-")
    if not safe:
        raise ExperimentLayoutError(f"{field} contains no filesystem-safe characters")
    if safe in {".", ".."}:
        raise ExperimentLayoutError(f"Invalid {field}: {value!r}")
    return safe


def normalize_experiment_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        raise ExperimentLayoutError("experiment number must not be empty")
    match = EXP_RE.fullmatch(text)
    if match:
        return f"exp{int(match.group('number'))}"
    if text.isdigit() and int(text) > 0:
        return f"exp{int(text)}"
    raise ExperimentLayoutError(
        "Experiment number must be a positive integer or expN (for example 12 or exp12)"
    )


def experiment_number(experiment_id: str) -> int:
    match = EXP_RE.fullmatch(str(experiment_id).strip().lower())
    if not match:
        raise ExperimentLayoutError(f"Invalid experiment ID: {experiment_id!r}")
    return int(match.group("number"))


def hierarchy_tokens(config: Dict[str, Any]) -> Tuple[str, ...]:
    ai = config.get("ai", {})
    fields = (
        ("family", ai.get("family")),
        ("architecture", ai.get("architecture")),
        ("variant", ai.get("variant")),
        ("application", ai.get("application")),
        ("dataset", ai.get("dataset")),
        ("framework", ai.get("framework")),
    )
    return tuple(safe_component(value, field) for field, value in fields)


def hierarchy_relative_path(config: Dict[str, Any]) -> Path:
    return Path(*hierarchy_tokens(config))


def branch_directory(root: str | Path, config: Dict[str, Any]) -> Path:
    return Path(root) / hierarchy_relative_path(config)


def existing_experiment_numbers(branch: str | Path) -> list[int]:
    path = Path(branch)
    if not path.exists():
        return []
    values: list[int] = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        match = EXP_RE.fullmatch(child.name.lower())
        if match:
            values.append(int(match.group("number")))
    return sorted(set(values))


def next_experiment_number(branch: str | Path) -> int:
    values = existing_experiment_numbers(branch)
    return (max(values) + 1) if values else 1


def locator_for(config: Dict[str, Any], experiment_id: str | None = None) -> str:
    exp_id = normalize_experiment_id(
        experiment_id or config.get("experiment", {}).get("experiment_id")
    )
    return str(hierarchy_relative_path(config) / exp_id)


def parse_locator(locator: str) -> Tuple[Tuple[str, ...], str]:
    raw = str(locator or "").strip().strip("/\\")
    parts = [part for part in re.split(r"[/\\]+", raw) if part]
    if len(parts) != 7:
        raise ExperimentLayoutError(
            "Experiment locator must contain exactly: "
            "family/architecture/variant/application/dataset/framework/expN"
        )
    tokens = tuple(safe_component(part, "locator component") for part in parts[:-1])
    exp_id = normalize_experiment_id(parts[-1])
    return tokens, exp_id


def apply_hierarchical_layout(
    config: Dict[str, Any],
    *,
    root: str | Path,
    experiment_id: str,
    role_token: str,
) -> Dict[str, Any]:
    exp_id = normalize_experiment_id(experiment_id)
    branch = branch_directory(root, config)
    run_root = branch / exp_id
    role_dir = run_root / safe_component(role_token, "role")
    experiment = config.setdefault("experiment", {})
    experiment.update(
        {
            "experiment_id": exp_id,
            "experiment_number": experiment_number(exp_id),
            "results_root": str(Path(root)),
            "branch_dir": str(branch),
            "run_dir": str(run_root),
            "storage_locator": str(hierarchy_relative_path(config) / exp_id),
            "output_dir": str(role_dir),
            "layout_version": "1.0",
        }
    )
    return config


def apply_proxy_locator_layout(
    config: Dict[str, Any],
    *,
    root: str | Path,
    locator: str,
) -> Dict[str, Any]:
    tokens, exp_id = parse_locator(locator)
    run_root = Path(root).joinpath(*tokens, exp_id)
    experiment = config.setdefault("experiment", {})
    experiment.update(
        {
            "experiment_id": exp_id,
            "experiment_number": experiment_number(exp_id),
            "results_root": str(Path(root)),
            "run_dir": str(run_root),
            "storage_locator": str(Path(*tokens, exp_id)),
            "output_dir": str(run_root / "proxy"),
            "layout_version": "1.0",
            # This is operator-supplied filesystem metadata only. The proxy
            # still receives no structured AI labels and these tokens are
            # never added to packet/feature predictor rows.
            "storage_locator_is_operator_metadata": True,
        }
    )
    return config


def materialize_role_metadata(config: Dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    output_dir = Path(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir_value = experiment.get("run_dir")
    if run_dir_value:
        (Path(run_dir_value) / "analysis").mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    role_manifest = {
        "experiment_id": experiment.get("experiment_id"),
        "experiment_number": experiment.get("experiment_number"),
        "storage_locator": experiment.get("storage_locator"),
        "output_dir": str(output_dir),
        "role": config.get("node", {}).get("role", "proxy"),
        "layout_version": experiment.get("layout_version", "legacy"),
    }
    if "ai" in config:
        ai = config.get("ai", {})
        role_manifest["hierarchy"] = {
            key: ai.get(key)
            for key in (
                "family",
                "architecture",
                "variant",
                "application",
                "dataset",
                "framework",
            )
        }
        run_dir_value = experiment.get("run_dir")
        if run_dir_value:
            run_dir = Path(run_dir_value)
            run_dir.mkdir(parents=True, exist_ok=True)
            common_manifest = {
                "experiment_id": experiment.get("experiment_id"),
                "experiment_number": experiment.get("experiment_number"),
                "storage_locator": experiment.get("storage_locator"),
                "layout_version": experiment.get("layout_version", "1.0"),
                "family": ai.get("family"),
                "architecture": ai.get("architecture"),
                "variant": ai.get("variant"),
                "application": ai.get("application"),
                "dataset": ai.get("dataset"),
                "framework": ai.get("framework"),
                "runtime": ai.get("runtime"),
                "task": config.get("execution", {}).get("task"),
                "deployment": config.get("execution", {}).get("deployment"),
                "transport": config.get("transport", {}).get("kind"),
            }
            with (run_dir / "experiment_manifest.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(common_manifest, handle, indent=2, sort_keys=True)
    with (output_dir / "role_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(role_manifest, handle, indent=2, sort_keys=True)


def write_role_status(
    config: Dict[str, Any],
    status: str,
    *,
    error: str | None = None,
) -> None:
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": config["experiment"].get("experiment_id"),
        "storage_locator": config["experiment"].get("storage_locator"),
        "role": config.get("node", {}).get("role", "proxy"),
        "status": str(status).upper(),
    }
    if error:
        payload["error"] = error
    with (output_dir / "experiment_status.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
