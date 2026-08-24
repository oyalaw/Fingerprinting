from __future__ import annotations

from typing import Any, Dict

from .client import ExperimentClient
from .experiment_output import enforce_experiment_output_policy
from .server import ExperimentServer


def run(config: Dict[str, Any]) -> None:
    role = config["node"]["role"]
    enforce_experiment_output_policy(
        config,
        role=role,
    )
    if role == "server":
        ExperimentServer(config).serve_forever()
    elif role == "client":
        ExperimentClient(config).run()
    else:
        raise ValueError(f"Unsupported role: {role}")
