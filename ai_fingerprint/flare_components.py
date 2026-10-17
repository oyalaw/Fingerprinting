"""
Server-side NVIDIA FLARE components used by flare_backend's FedAvg job.

WorkloadPersistor gives the FedAvg controller its initial global model: it builds the same
workload as the clients from the role config in $AIFP_FLARE_CONFIG and returns its weights as
numpy arrays (keys p00000, p00001, ... in parameter order, as flare_client_job.py expects).
"""
import os

import yaml

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor


class WorkloadPersistor(ModelPersistor):
    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        from ai_fingerprint.workloads import build_workload

        cfg = yaml.safe_load(open(os.environ["AIFP_FLARE_CONFIG"], encoding="utf-8"))
        arrays = build_workload(cfg).get_parameters()
        return make_model_learnable({f"p{i:05d}": a for i, a in enumerate(arrays)}, {})

    def save_model(self, model: ModelLearnable, fl_ctx: FLContext):
        # The global model is not needed after the run; traffic, not weights, is the measurement.
        pass
