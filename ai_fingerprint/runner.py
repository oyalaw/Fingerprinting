from __future__ import annotations

from typing import Any, Dict

from .client import ExperimentClient
from .server import ExperimentServer


def run(config: Dict[str, Any]) -> None:
    role = config["node"]["role"]
    if role == "server":
        ExperimentServer(config).serve_forever()
    elif role == "client":
        ExperimentClient(config).run()
    else:
        raise ValueError(f"Unsupported role: {role}")
