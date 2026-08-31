from __future__ import annotations

from typing import Any, Dict

from .client import ExperimentClient
from .experiment_layout import materialize_role_metadata, write_role_status
from .experiment_output import enforce_experiment_output_policy
from .server import ExperimentServer
from .result_collection import auto_upload_result_copy


def run(config: Dict[str, Any]) -> None:
    role = config["node"]["role"]
    enforce_experiment_output_policy(
        config,
        role=role,
    )
    materialize_role_metadata(config)
    write_role_status(config, "RUNNING")
    try:
        if role == "server":
            ExperimentServer(config).serve_forever()
        elif role == "client":
            ExperimentClient(config).run()
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
