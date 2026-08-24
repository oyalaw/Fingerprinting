from __future__ import annotations

import copy
import socket
import threading
import time

from ai_fingerprint.client import ExperimentClient
from ai_fingerprint.config import DEFAULT_CONFIG
from ai_fingerprint.server import ExperimentServer


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main() -> None:
    port = free_port()

    server_config = copy.deepcopy(DEFAULT_CONFIG)
    server_config["experiment"]["experiment_id"] = "SMOKE_SERVER"
    server_config["experiment"]["output_dir"] = "smoke_output"
    server_config["node"]["role"] = "server"
    server_config["node"]["host"] = "127.0.0.1"
    server_config["node"]["port"] = port
    server_config["ai"].update(
        {
            "family": "rnn",
            "architecture": "lstm",
            "variant": "lstm_2layer",
            "application": "activity_recognition",
            "dataset": "synthetic_sequence",
            "num_classes": 6,
            "sequence_length": 8,
            "input_dim": 4,
        }
    )
    server_config["execution"]["repetitions"] = 1
    server_config["execution"]["warmup"] = 0
    server_config["execution"]["interval_ms"] = 0

    client_config = copy.deepcopy(server_config)
    client_config["experiment"]["experiment_id"] = "SMOKE_CLIENT"
    client_config["node"]["role"] = "client"

    server = ExperimentServer(server_config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(1)

    client = ExperimentClient(client_config)
    client.run()
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
