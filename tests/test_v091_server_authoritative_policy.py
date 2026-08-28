from __future__ import annotations

import json
import socket
from pathlib import Path

from ai_fingerprint import config as config_module
from ai_fingerprint.client import ExperimentClient
from ai_fingerprint.federated_policy import (
    FederatedPolicyError,
    apply_training_policy,
    build_training_policy,
    validate_training_policy,
)
from ai_fingerprint.metadata import ground_truth_record
from ai_fingerprint.protocol import recv_frame
from ai_fingerprint.server import ExperimentServer


def _config(tmp_path: Path, role: str = "server"):
    cfg = config_module.deep_merge(config_module.DEFAULT_CONFIG, {})
    cfg["experiment"].update(
        {
            "experiment_id": "exp12",
            "output_dir": str(tmp_path / role),
        }
    )
    cfg["node"]["role"] = role
    cfg["execution"]["task"] = "training"
    cfg["execution"]["deployment"] = "federated"
    cfg["execution"]["batch_size"] = 7
    cfg["execution"]["learning_rate"] = 0.0025
    cfg["ai"]["input_size"] = 64
    cfg["federated"].update(
        {
            "rounds": 13,
            "local_epochs": 2,
            "steps_per_epoch": 11,
            "expected_clients": 2,
            "client_id": "client_2",
            "policy_source": "server",
            "policy_applied": role == "server",
        }
    )
    return cfg


def test_training_policy_contains_six_server_authoritative_controls(tmp_path: Path):
    cfg = _config(tmp_path)
    policy = build_training_policy(cfg)
    assert policy["input_size"] == 64
    assert policy["batch_size"] == 7
    assert policy["learning_rate"] == 0.0025
    assert policy["rounds"] == 13
    assert policy["local_epochs"] == 2
    assert policy["steps_per_epoch"] == 11
    assert policy["authority"] == "server"
    assert len(policy["policy_id"]) == 64


def test_client_applies_server_policy_over_local_placeholders(tmp_path: Path):
    server_cfg = _config(tmp_path, "server")
    client_cfg = _config(tmp_path, "client")
    client_cfg["ai"]["input_size"] = 224
    client_cfg["execution"]["batch_size"] = 1
    client_cfg["execution"]["learning_rate"] = 0.001
    client_cfg["federated"].update(
        {"rounds": 10, "local_epochs": 1, "steps_per_epoch": 10, "policy_applied": False}
    )

    policy = build_training_policy(server_cfg)
    apply_training_policy(client_cfg, policy)

    assert client_cfg["ai"]["input_size"] == 64
    assert client_cfg["execution"]["batch_size"] == 7
    assert client_cfg["execution"]["learning_rate"] == 0.0025
    assert client_cfg["federated"]["rounds"] == 13
    assert client_cfg["federated"]["local_epochs"] == 2
    assert client_cfg["federated"]["steps_per_epoch"] == 11
    assert client_cfg["federated"]["policy_source"] == "server"
    assert client_cfg["federated"]["policy_applied"] is True


def test_policy_digest_detects_tampering(tmp_path: Path):
    policy = build_training_policy(_config(tmp_path))
    policy["batch_size"] = 99
    try:
        validate_training_policy(policy, expected_experiment_id="exp12")
    except FederatedPolicyError as exc:
        assert "digest" in str(exc).lower()
    else:
        raise AssertionError("tampered policy was accepted")


def test_server_policy_handler_returns_authoritative_policy(tmp_path: Path):
    cfg = _config(tmp_path, "server")
    server = ExperimentServer(cfg)
    left, right = socket.socketpair()
    try:
        server._handle_fl_policy_get(
            left,
            "req1",
            {"experiment_id": "exp12", "client_id": "client_2"},
        )
        header, payload = recv_frame(right)
        assert payload == b""
        assert header["status"] == "ok"
        assert header["training_policy"]["rounds"] == 13
        assert header["training_policy"]["batch_size"] == 7
    finally:
        left.close()
        right.close()


def test_federated_client_defers_generator_until_policy(tmp_path: Path):
    cfg = _config(tmp_path, "client")
    client = ExperimentClient(cfg)
    assert client.generator is None


def test_pre_handshake_ground_truth_does_not_claim_placeholder_policy(tmp_path: Path):
    cfg = _config(tmp_path, "client")
    cfg["federated"]["policy_applied"] = False
    record = ground_truth_record(cfg)
    assert record["input_size"] is None
    assert record["batch_size"] is None
    assert record["training_policy_status"] == "pending_server"


def test_interactive_federated_client_does_not_prompt_for_server_policy_controls(
    monkeypatch, tmp_path: Path
):
    import ai_fingerprint.cli as cli

    prompts: list[str] = []

    def fake_choose(prompt, options):
        prompts.append(prompt)
        wanted = {
            "Select experiment task": "training",
            "Select deployment": "federated",
            "Select framework": "pytorch",
            "Select execution runtime": "native",
            "Select model family": "autoencoder",
            "Select architecture": "convolutional_autoencoder",
            "Select architecture variant": "convolutional_autoencoder_2layer",
            "Select application": "reconstruction",
            "Select dataset": "synthetic_image",
            "Select device label": "dell_desktop",
            "Select operating system": "ubuntu",
            "Select network transport": "tcp",
        }
        value = wanted.get(prompt, options[0])
        assert value in options, (prompt, value, options)
        return value

    def fake_text(prompt, default=""):
        prompts.append(prompt)
        if prompt == "Experiment results root":
            return str(tmp_path)
        if prompt == "Federated client ID":
            return "client_2"
        return str(default)

    def fake_int(prompt, default=None):
        prompts.append(prompt)
        return int(default)

    def fake_yes_no(prompt, default=True):
        prompts.append(prompt)
        if prompt == "Enable client/server resource telemetry":
            return False
        if prompt == "Log per-round training performance":
            return False
        return bool(default)

    monkeypatch.setattr(cli, "choose", fake_choose)
    monkeypatch.setattr(cli, "ask_text", fake_text)
    monkeypatch.setattr(cli, "ask_int", fake_int)
    monkeypatch.setattr(cli, "ask_yes_no", fake_yes_no)

    cfg = cli.interactive_configure(forced_role="client")
    assert cfg["federated"]["client_id"] == "client_2"
    for forbidden_prompt in (
        "Image input size",
        "Batch size",
        "Learning rate",
        "Federated rounds",
        "Local epochs per round",
        "Local steps per epoch",
    ):
        assert forbidden_prompt not in prompts


def test_interactive_federated_server_prompts_for_authoritative_policy_controls(
    monkeypatch, tmp_path: Path
):
    import ai_fingerprint.cli as cli

    prompts: list[str] = []

    def fake_choose(prompt, options):
        prompts.append(prompt)
        wanted = {
            "Select experiment task": "training",
            "Select deployment": "federated",
            "Select framework": "pytorch",
            "Select execution runtime": "native",
            "Select model family": "autoencoder",
            "Select architecture": "convolutional_autoencoder",
            "Select architecture variant": "convolutional_autoencoder_2layer",
            "Select application": "reconstruction",
            "Select dataset": "synthetic_image",
            "Select device label": "jetson_agx_orin",
            "Select operating system": "ubuntu",
            "Select network transport": "tcp",
        }
        value = wanted.get(prompt, options[0])
        assert value in options, (prompt, value, options)
        return value

    def fake_text(prompt, default=""):
        prompts.append(prompt)
        if prompt == "Experiment results root":
            return str(tmp_path)
        return str(default)

    def fake_int(prompt, default=None):
        prompts.append(prompt)
        return int(default)

    def fake_yes_no(prompt, default=True):
        prompts.append(prompt)
        if prompt == "Enable client/server resource telemetry":
            return False
        if prompt == "Log per-round training performance":
            return False
        return bool(default)

    monkeypatch.setattr(cli, "choose", fake_choose)
    monkeypatch.setattr(cli, "ask_text", fake_text)
    monkeypatch.setattr(cli, "ask_int", fake_int)
    monkeypatch.setattr(cli, "ask_yes_no", fake_yes_no)

    cli.interactive_configure(forced_role="server")
    for required_prompt in (
        "Image input size",
        "Batch size",
        "Learning rate",
        "Federated rounds",
        "Local epochs per round",
        "Local steps per epoch",
    ):
        assert required_prompt in prompts
