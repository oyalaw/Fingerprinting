from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .metadata import output_role_token


class ExistingExperimentError(RuntimeError):
    pass


VALID_EXISTING_OUTPUT_POLICIES = {
    "error",
    "archive",
}


def _safe_experiment_id(value: Any) -> str:
    experiment_id = str(value or "").strip()
    if not experiment_id:
        raise ExistingExperimentError(
            "experiment_id must not be empty"
        )
    if "/" in experiment_id or "\\" in experiment_id:
        raise ExistingExperimentError(
            "experiment_id must not contain path separators"
        )
    return experiment_id


def _client_server_candidates(
    output_dir: Path,
    experiment_id: str,
    role: str,
    role_token: str | None = None,
) -> List[Path]:
    # Restrict the collision domain to the current role/client identity. The
    # same experiment_id is intentionally shared by client, server, and proxy.
    token = role_token or role
    prefix = f"{experiment_id}_{token}_"
    return sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix)
    ) if output_dir.exists() else []


def _proxy_candidates(
    output_dir: Path,
    experiment_id: str,
) -> List[Path]:
    if not output_dir.exists():
        return []

    exact_names = {
        f"{experiment_id}.pcap",
        f"{experiment_id}.pcapng",
        f"{experiment_id}_packet_sequence.csv",
        f"{experiment_id}_fingerprint_sequence.csv",
        f"{experiment_id}_features.csv",
        f"{experiment_id}_manifest.json",
        f"{experiment_id}_proxy_forwarding.csv",
        f"{experiment_id}_proxy_summary.json",
    }

    return sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and (
            path.name in exact_names
            or path.name.startswith(f"{experiment_id}_proxy_")
            or path.name.startswith(f"{experiment_id}__")
        )
    )


def find_existing_outputs(
    config: Dict[str, Any],
    role: str | None = None,
) -> List[Path]:
    experiment = config.get("experiment", {})
    experiment_id = _safe_experiment_id(
        experiment.get("experiment_id")
    )
    output_dir = Path(
        experiment.get("output_dir", "experiments/results")
    )

    resolved_role = role
    if resolved_role is None:
        node = config.get("node", {})
        resolved_role = str(node.get("role", "")).strip()

    if resolved_role in {"client", "server"}:
        token = output_role_token(config)
        return _client_server_candidates(
            output_dir,
            experiment_id,
            resolved_role,
            role_token=token,
        )
    if resolved_role == "proxy":
        return _proxy_candidates(
            output_dir,
            experiment_id,
        )

    raise ExistingExperimentError(
        f"Unsupported experiment role for output protection: "
        f"{resolved_role!r}"
    )


def archive_existing_outputs(
    config: Dict[str, Any],
    role: str | None = None,
    paths: Iterable[Path] | None = None,
) -> Path | None:
    experiment = config["experiment"]
    experiment_id = _safe_experiment_id(
        experiment["experiment_id"]
    )
    output_dir = Path(experiment["output_dir"])
    resolved_role = role or str(
        config.get("node", {}).get("role", "")
    )

    existing = list(
        paths
        if paths is not None
        else find_existing_outputs(
            config,
            role=resolved_role,
        )
    )
    if not existing:
        return None

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_dir = (
        output_dir
        / "_archive"
        / experiment_id
        / resolved_role
        / stamp
    )
    archive_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in existing:
        shutil.move(
            str(path),
            str(archive_dir / path.name),
        )

    return archive_dir


def enforce_experiment_output_policy(
    config: Dict[str, Any],
    role: str | None = None,
) -> None:
    """
    Protect experimental outputs before workload/capture startup.

    Default behavior is fail-closed: an existing output file for the same
    experiment_id and role causes the run to stop. Set
    experiment.existing_output_policy=archive to move the old role-specific
    files into an archive directory before starting a fresh run.
    """
    experiment = config.setdefault("experiment", {})
    policy = str(
        experiment.get("existing_output_policy", "error")
    ).strip().lower()

    if policy not in VALID_EXISTING_OUTPUT_POLICIES:
        raise ExistingExperimentError(
            "experiment.existing_output_policy must be one of "
            f"{sorted(VALID_EXISTING_OUTPUT_POLICIES)}"
        )

    existing = find_existing_outputs(
        config,
        role=role,
    )
    if not existing:
        return

    resolved_role = role or str(
        config.get("node", {}).get("role", "")
    )
    experiment_id = str(
        experiment["experiment_id"]
    )

    if policy == "archive":
        archive_dir = archive_existing_outputs(
            config,
            role=resolved_role,
            paths=existing,
        )
        print(
            f"[experiment] archived {len(existing)} existing "
            f"{resolved_role} output file(s) for "
            f"experiment {experiment_id!r} to {archive_dir}"
        )
        return

    listing = "\n".join(
        f"  - {path}"
        for path in existing
    )
    raise ExistingExperimentError(
        f"Experiment ID {experiment_id!r} already has "
        f"{resolved_role} output files:\n"
        f"{listing}\n"
        "Refusing to append or mix multiple runs. "
        "Choose a new experiment ID or explicitly archive the "
        "existing run first."
    )
