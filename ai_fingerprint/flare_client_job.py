"""
Client training script run by NVIDIA FLARE on each site (via ScriptRunner).

It reads the role config from $AIFP_FLARE_CONFIG (written by FlareClient), builds the same data
partition and model as the custom backend, and for every round received from the FLARE server
loads the global weights, trains local_epochs x steps_per_epoch batches, logs the round to
round_metrics.csv, and sends the updated weights back.
"""
import os
import time
from pathlib import Path

import numpy as np
import yaml

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

from ai_fingerprint.datasets import InputGenerator
from ai_fingerprint.experiment_integrity import utc_now_iso
from ai_fingerprint.flower_backend import ROUND_FIELDS, _append, _labels, _nbytes
from ai_fingerprint.workloads import build_workload


def to_dict(arrays):
    return {f"p{i:05d}": a for i, a in enumerate(arrays)}


def from_dict(params):
    return [np.asarray(params[k]) for k in sorted(params)]


def main():
    cfg = yaml.safe_load(open(os.environ["AIFP_FLARE_CONFIG"], encoding="utf-8"))
    fed = cfg["federated"]
    client_id = str(fed["client_id"])
    generator = InputGenerator(cfg)
    workload = build_workload(cfg)
    epochs, steps = int(fed["local_epochs"]), int(fed["steps_per_epoch"])
    out = Path(cfg["experiment"]["output_dir"]) / "round_metrics.csv"
    labels = _labels(cfg, "nvidia_flare")

    flare.init()
    while flare.is_running():
        model = flare.receive()
        if model is None:
            break
        start_utc = utc_now_iso()
        if model.params:
            workload.set_parameters(from_dict(model.params))
        t0 = time.perf_counter()
        losses, examples = [], 0
        for _ in range(epochs):
            for _ in range(steps):
                x, y = generator.training_batch()
                losses.append(float(workload.train_batch(x, y)["loss"]))
                examples += int(x.shape[0])
        train_sec = time.perf_counter() - t0
        params = workload.get_parameters()
        _append(out, ROUND_FIELDS, {
            **labels, "client_id": client_id, "round": model.current_round, "train_samples": examples,
            "train_loss": float(np.mean(losses)) if losses else None, "training_time_sec": train_sec,
            "model_size_bytes": _nbytes(params), "timestamp_start_utc": start_utc,
            "timestamp_end_utc": utc_now_iso()})
        flare.send(FLModel(params=to_dict(params), params_type=ParamsType.FULL,
                           meta={"NUM_STEPS_CURRENT_ROUND": examples}))


if __name__ == "__main__":
    main()
