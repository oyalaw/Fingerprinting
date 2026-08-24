from __future__ import annotations

import copy
import csv
import json
import time

from ai_fingerprint.config import DEFAULT_CONFIG
from ai_fingerprint.resource_monitor import RESOURCE_FIELDS, ResourceMonitor


def test_resource_monitor_writes_csv_and_summary(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["experiment_id"] = "RESOURCE_TEST"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["node"]["role"] = "client"
    config["device"]["label"] = "test_device"
    config["resource_monitor"].update(
        {
            "enabled": True,
            "interval_ms": 100,
            "network_interface": None,
            "gpu_index": 0,
            "power_enabled": False,
        }
    )

    monitor = ResourceMonitor(config)
    monitor.start()

    # Generate a small amount of CPU and memory activity.
    values = [index * index for index in range(10000)]
    assert values[10] == 100

    time.sleep(0.25)
    summary = monitor.stop()

    assert monitor.csv_path.exists()
    assert monitor.summary_path.exists()
    assert summary["experiment_id"] == "RESOURCE_TEST"
    assert summary["sample_count"] >= 2
    assert summary["bytes_sent"] >= 0
    assert summary["bytes_received"] >= 0

    with monitor.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) >= 2
    assert set(RESOURCE_FIELDS).issubset(rows[0].keys())
    assert rows[0]["role"] == "client"
    assert rows[0]["device"] == "test_device"

    saved_summary = json.loads(
        monitor.summary_path.read_text(encoding="utf-8")
    )
    assert saved_summary["schema_version"] == "1.0"


def test_resource_monitor_disabled(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["experiment"]["experiment_id"] = "RESOURCE_DISABLED"
    config["experiment"]["output_dir"] = str(tmp_path)
    config["resource_monitor"]["enabled"] = False

    monitor = ResourceMonitor(config)
    monitor.start()
    summary = monitor.stop()

    assert summary == {}
    assert not monitor.csv_path.exists()
