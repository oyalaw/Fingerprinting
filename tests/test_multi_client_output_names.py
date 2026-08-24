from __future__ import annotations

import copy

from ai_fingerprint.config import DEFAULT_CONFIG
from ai_fingerprint.metadata import EventLogger
from ai_fingerprint.resource_monitor import ResourceMonitor


def _federated_client_config(tmp_path, client_id):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["experiment_id"] = "EXP_MULTI"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["node"]["role"] = "client"
    config["execution"]["task"] = "training"
    config["execution"]["deployment"] = "federated"
    config["federated"]["client_id"] = client_id
    config["resource_monitor"]["enabled"] = False
    return config


def test_federated_clients_get_distinct_ground_truth_filenames(tmp_path):
    logger1 = EventLogger(_federated_client_config(tmp_path, "client_1"))
    logger2 = EventLogger(_federated_client_config(tmp_path, "client_2"))
    assert logger1.path.name == "EXP_MULTI_client_1_ground_truth.jsonl"
    assert logger2.path.name == "EXP_MULTI_client_2_ground_truth.jsonl"
    assert logger1.path != logger2.path


def test_numeric_client_id_is_normalized_in_resource_filename(tmp_path):
    monitor = ResourceMonitor(_federated_client_config(tmp_path, "2"))
    assert monitor.csv_path.name == "EXP_MULTI_client_2_resource.csv"
    assert monitor.summary_path.name == "EXP_MULTI_client_2_resource_summary.json"
