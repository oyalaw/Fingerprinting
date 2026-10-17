"""
Server-side NVIDIA FLARE components used by flare_backend's FedAvg job.

LoggingFedAvg is FLARE's FedAvg plus one row per client update, written as each update arrives
(client_update_metrics.csv in the server's output folder, like the custom backend).

WorkloadPersistor gives the FedAvg controller its initial global model: it builds the same
workload as the clients from the role config in $AIFP_FLARE_CONFIG and returns its weights as
numpy arrays (keys p00000, p00001, ... in parameter order, as flare_client_job.py expects).
"""
import os

import yaml

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.workflows.fedavg import FedAvg

UPDATE_FIELDS = [
    "experiment_id", "round", "client_id", "fl_framework", "num_examples", "payload_bytes",
    "client_send_utc", "receive_timestamp_utc",
]


class LoggingFedAvg(FedAvg):
    def _process_result(self, client_task, fl_ctx) -> None:
        from pathlib import Path

        from ai_fingerprint.experiment_integrity import utc_now_iso
        from ai_fingerprint.flower_backend import _append

        received = utc_now_iso()
        before = len(self._results)
        super()._process_result(client_task, fl_ctx)
        try:
            cfg = yaml.safe_load(open(os.environ["AIFP_FLARE_CONFIG"], encoding="utf-8"))
            model = self._results[-1] if len(self._results) > before else None
            meta = (model.meta or {}) if model is not None else {}
            size = sum(getattr(v, "nbytes", 0) for v in (model.params or {}).values()) if model is not None else None
            _append(Path(cfg["experiment"]["output_dir"]) / "client_update_metrics.csv", UPDATE_FIELDS, {
                "experiment_id": cfg["experiment"]["experiment_id"], "round": self.current_round,
                "client_id": meta.get("client_id") or client_task.client.name, "fl_framework": "nvidia_flare",
                "num_examples": meta.get("NUM_STEPS_CURRENT_ROUND"), "payload_bytes": size,
                "client_send_utc": meta.get("client_send_utc"), "receive_timestamp_utc": received})
        except Exception as exc:  # logging must never break the FL job
            self.warning(f"client update logging failed: {exc}")


class WorkloadPersistor(ModelPersistor):
    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        from ai_fingerprint.workloads import build_workload

        cfg = yaml.safe_load(open(os.environ["AIFP_FLARE_CONFIG"], encoding="utf-8"))
        arrays = build_workload(cfg).get_parameters()
        return make_model_learnable({f"p{i:05d}": a for i, a in enumerate(arrays)}, {})

    def save_model(self, model: ModelLearnable, fl_ctx: FLContext):
        # The global model is not needed after the run; traffic, not weights, is the measurement.
        pass
