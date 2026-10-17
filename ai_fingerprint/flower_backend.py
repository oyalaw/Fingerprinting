"""
Flower (flwr) backend: the same models, data partitions, FedAvg, monitoring and output layout as
the custom backend, with Flower's gRPC transport on the wire.

Selected with `federated.fl_framework: flower`. The server runs `flwr.server.start_server` with a
FedAvg strategy that waits for all `federated.expected_clients` every round; clients run
`flwr.client.start_client` against `node.host:node.port`, which is the proxy address, so the
traffic passes the observation point exactly as with the custom backend. Flower carries the model
in protobuf messages over one long-lived HTTP/2 connection per client (insecure, or TLS when
`transport.kind: tls` with certfile/keyfile/cafile).

Differences from the custom backend
  * no server-to-client policy exchange: every role reads the training policy from its own config
    (the campaign planner writes identical policies), and the client's data partition index
    follows its client ID (client_1 -> 0, client_2 -> 1, ...)
  * per-round client timing covers local training only; download and upload happen inside gRPC
  * the client's source port is not known to the backend, so ground truth records its IP only

Requires the optional package: pip install "flwr==1.13.1"
"""
from __future__ import annotations

import csv
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .datasets import InputGenerator
from .experiment_coordination import ActiveRunCoordinator
from .experiment_integrity import utc_now_iso
from .experiment_layout import materialize_role_metadata
from .metadata import EventLogger
from .resource_monitor import ResourceMonitor
from .workloads import build_workload

ROUND_FIELDS = [
    "experiment_id", "client_id", "round", "family", "architecture", "variant", "application",
    "dataset", "framework", "fl_framework", "global_rounds", "local_epochs", "steps_per_epoch",
    "train_samples", "train_loss", "training_time_sec", "model_size_bytes",
    "timestamp_start_utc", "timestamp_end_utc",
]
SERVER_FIELDS = [
    "experiment_id", "round", "fl_framework", "clients", "aggregation_time_ms",
    "model_size_bytes", "timestamp_utc",
]
# One row per client update on the server, like the custom backend's client_update_metrics.csv.
# Flower hands the strategy all of a round's results together, so the server time is when the
# round's aggregation began (after the last update arrived); client_send_utc is per client.
UPDATE_FIELDS = [
    "experiment_id", "round", "client_id", "fl_framework", "num_examples", "payload_bytes",
    "train_loss", "client_send_utc", "server_aggregate_utc",
]


def _flwr():
    try:
        import flwr
    except ImportError as exc:
        raise RuntimeError('federated.fl_framework=flower requires: pip install "flwr==1.13.1"') from exc
    return flwr


def _append(path: Path, fields: List[str], row: Dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _nbytes(arrays: List[np.ndarray]) -> int:
    return int(sum(a.nbytes for a in arrays))


def _certificates(config: Dict[str, Any]) -> Optional[Tuple[bytes, bytes, bytes]]:
    t = config.get("transport", {}) or {}
    if str(t.get("kind", "tcp")).lower() != "tls":
        return None
    paths = [t.get("cafile"), t.get("certfile"), t.get("keyfile")]
    missing = [p for p in paths if not p or not Path(p).exists()]
    if missing:
        raise RuntimeError(
            "Flower over TLS needs the testbed certificates (transport.cafile, certfile, keyfile); "
            f"missing: {missing}. Run `main.py tls-provision` and copy tls/ to every machine, "
            "or set transport.kind: tcp to run unencrypted.")
    return tuple(Path(p).read_bytes() for p in paths)  # type: ignore[return-value]


def _local_ip_towards(host: str, port: int) -> Optional[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((host, port))
            return s.getsockname()[0]
    except OSError:
        return None


def _labels(config: Dict[str, Any], fl_framework: str = "flower") -> Dict[str, Any]:
    ai, fed = config["ai"], config["federated"]
    return {"experiment_id": config["experiment"]["experiment_id"], "family": ai["family"],
            "architecture": ai["architecture"], "variant": ai["variant"], "application": ai["application"],
            "dataset": ai["dataset"], "framework": ai["framework"], "fl_framework": fl_framework,
            "global_rounds": int(fed["rounds"]), "local_epochs": int(fed["local_epochs"]),
            "steps_per_epoch": int(fed["steps_per_epoch"])}


class FlowerServer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = EventLogger(config)
        self.out = Path(config["experiment"]["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)

    def serve_forever(self) -> None:
        flwr = _flwr()
        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
        cfg, fed = self.config, self.config["federated"]
        coordinator = None
        if bool((cfg.get("coordination", {}) or {}).get("enabled", True)):
            coordinator = ActiveRunCoordinator(cfg)
            coordinator.start()
            print(f"[server] experiment coordinator: {coordinator.host}:{coordinator.port} "
                  f"run_id={coordinator.run_id}")
        materialize_role_metadata(cfg)
        self.logger.refresh_base(cfg)
        initial = build_workload(cfg).get_parameters()
        n = int(fed["expected_clients"])
        rounds = int(fed["rounds"])
        metrics_csv = self.out / "round_metrics.csv"
        updates_csv = self.out / "client_update_metrics.csv"
        exp_id = cfg["experiment"]["experiment_id"]
        logger = self.logger

        class LoggedFedAvg(flwr.server.strategy.FedAvg):
            def aggregate_fit(self, server_round, results, failures):
                agg_utc = utc_now_iso()
                for _, fit_res in results:
                    m = fit_res.metrics or {}
                    _append(updates_csv, UPDATE_FIELDS, {
                        "experiment_id": exp_id, "round": server_round - 1,
                        "client_id": m.get("client_id"), "fl_framework": "flower",
                        "num_examples": fit_res.num_examples,
                        "payload_bytes": _nbytes(parameters_to_ndarrays(fit_res.parameters)),
                        "train_loss": m.get("train_loss"), "client_send_utc": m.get("client_send_utc"),
                        "server_aggregate_utc": agg_utc})
                t0 = time.perf_counter_ns()
                agg = super().aggregate_fit(server_round, results, failures)
                ms = (time.perf_counter_ns() - t0) / 1e6
                size = _nbytes(parameters_to_ndarrays(agg[0])) if agg and agg[0] is not None else None
                _append(metrics_csv, SERVER_FIELDS, {
                    "experiment_id": exp_id, "round": server_round - 1, "fl_framework": "flower",
                    "clients": len(results), "aggregation_time_ms": round(ms, 3),
                    "model_size_bytes": size, "timestamp_utc": utc_now_iso()})
                logger.write("federated_round_aggregated", round=server_round - 1, clients=len(results),
                             failures=len(failures), aggregation_time_ms=ms, fl_framework="flower")
                return agg

        strategy = LoggedFedAvg(
            fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=n, min_evaluate_clients=0,
            min_available_clients=n, initial_parameters=ndarrays_to_parameters(initial),
            on_fit_config_fn=lambda r: {"server_round": r},
        )
        address = f"{cfg['node']['host']}:{int(cfg['node']['port'])}"
        print(f"[server] Flower listening on {address}: {rounds} rounds, waiting for {n} clients")
        monitor = ResourceMonitor(cfg)
        self.logger.write("experiment_start", fl_framework="flower", rounds=rounds, expected_clients=n)
        monitor.start()
        try:
            flwr.server.start_server(
                server_address=address, config=flwr.server.ServerConfig(num_rounds=rounds),
                strategy=strategy, grpc_max_message_length=2_000_000_000,
                certificates=_certificates(cfg))
        finally:
            monitor.stop()
            self.logger.write("experiment_stop", fl_framework="flower")
            if coordinator is not None:
                coordinator.stop()


class FlowerClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = EventLogger(config)
        self.out = Path(config["experiment"]["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        flwr = _flwr()
        cfg, fed = self.config, self.config["federated"]
        client_id = str(fed["client_id"])
        part = cfg.setdefault("data", {}).setdefault("partition", {})
        try:
            part["client_index"] = int(client_id.rsplit("_", 1)[1]) - 1
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"client_id {client_id!r} must look like client_<n>") from exc
        part["client_count"] = int(fed["expected_clients"])
        part["client_id"] = client_id
        if part.get("seed") is None:
            part["seed"] = int(cfg.get("execution", {}).get("seed", 42))
        materialize_role_metadata(cfg)
        self.logger.refresh_base(cfg)

        generator = InputGenerator(cfg)
        workload = build_workload(cfg)
        epochs, steps = int(fed["local_epochs"]), int(fed["steps_per_epoch"])
        metrics_csv = self.out / "round_metrics.csv"
        labels = _labels(cfg)
        logger = self.logger
        host, port = str(cfg["node"]["host"]), int(cfg["node"]["port"])
        logger.write("network_registration", client_id=client_id, local_ip=_local_ip_towards(host, port),
                     local_port=None, remote_host=host, remote_port=port, transport="grpc", fl_framework="flower")

        class Client(flwr.client.NumPyClient):
            def get_parameters(self, config):
                return workload.get_parameters()

            def fit(self, parameters, config):
                start_utc = utc_now_iso()
                rnd = int(config.get("server_round", 0)) - 1
                workload.set_parameters(parameters)
                t0 = time.perf_counter()
                losses, examples = [], 0
                for _ in range(epochs):
                    for _ in range(steps):
                        x, y = generator.training_batch()
                        losses.append(float(workload.train_batch(x, y)["loss"]))
                        examples += int(x.shape[0])
                train_sec = time.perf_counter() - t0
                params = workload.get_parameters()
                _append(metrics_csv, ROUND_FIELDS, {
                    **labels, "client_id": client_id, "round": rnd, "train_samples": examples,
                    "train_loss": float(np.mean(losses)) if losses else None,
                    "training_time_sec": train_sec, "model_size_bytes": _nbytes(params),
                    "timestamp_start_utc": start_utc, "timestamp_end_utc": utc_now_iso()})
                logger.write("federated_phase", round=rnd, client_id=client_id, phase="Training",
                             phase_time_ms=train_sec * 1000.0, fl_framework="flower")
                return params, examples, {"train_loss": float(np.mean(losses)) if losses else 0.0,
                                          "client_id": client_id, "client_send_utc": utc_now_iso()}

        address = f"{host}:{port}"
        certs = _certificates(cfg)
        print(f"[client] {client_id} connecting to Flower server via {address}")
        monitor = ResourceMonitor(cfg)
        logger.write("experiment_start", fl_framework="flower")
        monitor.start()
        try:
            flwr.client.start_client(
                server_address=address, client=Client().to_client(),
                grpc_max_message_length=2_000_000_000,
                insecure=certs is None, root_certificates=certs[0] if certs else None,
                max_retries=60, max_wait_time=600)
        finally:
            monitor.stop()
            logger.write("experiment_stop", fl_framework="flower")
