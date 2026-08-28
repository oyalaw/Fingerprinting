from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


CONTRACT_VERSION = "1.0"


def build_model_contract(
    config: Dict[str, Any],
    parameters: Iterable[np.ndarray],
) -> Dict[str, Any]:
    ai = config.get("ai", {})
    execution = config.get("execution", {})
    tensors = []
    for index, value in enumerate(parameters):
        array = np.asarray(value)
        tensors.append(
            {
                "index": index,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    core = {
        "contract_version": CONTRACT_VERSION,
        # Salt the on-wire contract hash with the coordinated experiment ID.
        # This prevents the hash itself from becoming a stable cross-run model
        # identifier while still allowing client/server compatibility checks.
        "experiment_id": config.get("experiment", {}).get("experiment_id"),
        "framework": ai.get("framework"),
        "runtime": ai.get("runtime"),
        "family": ai.get("family"),
        "architecture": ai.get("architecture"),
        "variant": ai.get("variant"),
        "application": ai.get("application"),
        "input_size": ai.get("input_size"),
        "input_dim": ai.get("input_dim"),
        "sequence_length": ai.get("sequence_length"),
        "vocab_size": ai.get("vocab_size"),
        "max_text_length": ai.get("max_text_length"),
        "num_classes": ai.get("num_classes"),
        "precision": execution.get("precision"),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    core["contract_id"] = hashlib.sha256(canonical).hexdigest()
    return core


def compact_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    # Keep model labels and tensor shapes local. Only a run-salted digest and
    # tensor count cross the FL channel. The proxy never parses these fields.
    return {
        "contract_version": contract.get("contract_version"),
        "contract_id": contract.get("contract_id"),
        "tensor_count": contract.get("tensor_count"),
    }


def write_model_contract(config: Dict[str, Any], contract: Dict[str, Any]) -> Path:
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "model_contract.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
    return target
