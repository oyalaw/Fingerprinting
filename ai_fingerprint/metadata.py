from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ground_truth_record(config: Dict[str, Any]) -> Dict[str, Any]:
    ai = config["ai"]
    data = config["data"]
    return {
        "experiment_id": config["experiment"]["experiment_id"],
        "role": config["node"]["role"],
        "framework": ai["framework"],
        "runtime": ai["runtime"],
        "family": ai["family"],
        "architecture": ai["architecture"],
        "variant": ai["variant"],
        "application": ai["application"],
        "dataset": ai["dataset"],
        "dataset_split": data["split"],
        "device": config["device"]["label"],
        "operating_system": config["device"]["operating_system"],
        "task": config["execution"]["task"],
        "deployment": config["execution"]["deployment"],
        "execution_mode": (
            f"{config['execution']['task']}_"
            f"{config['execution']['deployment']}"
        ),
        "precision": config["execution"]["precision"],
        "batch_size": config["execution"]["batch_size"],
        "input_size": ai["input_size"],
    }


class EventLogger:
    def __init__(self, config: Dict[str, Any]) -> None:
        output_dir = Path(config["experiment"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        experiment_id = config["experiment"]["experiment_id"]
        role = config["node"]["role"]
        self.path = output_dir / f"{experiment_id}_{role}_ground_truth.jsonl"
        self.base = ground_truth_record(config)

    def write(self, event: str, **fields: Any) -> None:
        record = dict(self.base)
        record.update(
            {
                "event": event,
                "timestamp_utc": utc_now_iso(),
            }
        )
        record.update(fields)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
