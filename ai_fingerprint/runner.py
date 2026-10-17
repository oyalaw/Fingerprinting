from __future__ import annotations

import os
from typing import Any, Dict

from .client import ExperimentClient
from .experiment_layout import materialize_role_metadata, write_role_status
from .experiment_output import enforce_experiment_output_policy
from .server import ExperimentServer
from .result_collection import auto_upload_result_copy
from .host_environment import device_preflight


def run(config: Dict[str, Any]) -> None:
    role = config["node"]["role"]
    # Refuse to start a config on the wrong machine (set AIFP_SKIP_DEVICE_CHECK=1 to override).
    problems = device_preflight(config)
    if problems and os.environ.get("AIFP_SKIP_DEVICE_CHECK") != "1":
        raise RuntimeError("Device check failed: " + "; ".join(problems))
    enforce_experiment_output_policy(
        config,
        role=role,
    )
    materialize_role_metadata(config)
    write_role_status(config, "RUNNING")
    try:
        server_cls, client_cls = ExperimentServer, ExperimentClient
        if config.get("execution", {}).get("deployment") == "federated":
            fl_framework = str(config.get("federated", {}).get("fl_framework", "custom_fedavg_tcp"))
            if fl_framework == "flower":
                from .flower_backend import FlowerClient as client_cls, FlowerServer as server_cls
            elif fl_framework == "nvidia_flare":
                from .flare_backend import FlareClient as client_cls, FlareServer as server_cls
        if role == "server":
            server_cls(config).serve_forever()
        elif role == "client":
            client_cls(config).run()
        else:
            raise ValueError(f"Unsupported role: {role}")
    except KeyboardInterrupt:
        write_role_status(config, "STOPPED")
        raise
    except Exception as exc:
        write_role_status(
            config,
            "FAILED",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        write_role_status(config, "COMPLETE")
        auto_upload_result_copy(config)
