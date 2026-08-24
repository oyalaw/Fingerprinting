from __future__ import annotations

import socket
import ssl
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .federated import SynchronousFedAvgCoordinator
from .metadata import EventLogger
from .protocol import (
    array_to_bytes,
    arrays_to_bytes,
    bytes_to_array,
    bytes_to_arrays,
    bytes_to_training_batch,
    recv_frame,
    send_frame,
)
from .resource_monitor import ResourceMonitor
from .workloads import build_workload


class ExperimentServer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = EventLogger(config)
        self.workload = None
        self.stop_event = threading.Event()

        self._workload_lock = threading.Lock()
        self._federated_lock = threading.Lock()
        self._federated = None

    def _create_listener(self) -> socket.socket:
        host = self.config["node"]["host"]
        port = int(self.config["node"]["port"])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        sock.bind((host, port))
        sock.listen(16)
        return sock

    def _wrap_tls(self, conn: socket.socket) -> socket.socket:
        transport = self.config["transport"]
        if transport["kind"] != "tls":
            return conn

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            certfile=transport["certfile"],
            keyfile=transport["keyfile"],
        )
        return context.wrap_socket(
            conn,
            server_side=True,
        )

    def _ensure_workload(self):
        with self._workload_lock:
            if self.workload is None:
                self.workload = build_workload(self.config)
            return self.workload

    def _ensure_federated_coordinator(
        self,
    ) -> SynchronousFedAvgCoordinator:
        with self._federated_lock:
            if self._federated is None:
                workload = self._ensure_workload()
                fed = self.config["federated"]
                self._federated = SynchronousFedAvgCoordinator(
                    workload=workload,
                    rounds=int(fed["rounds"]),
                    expected_clients=int(
                        fed["expected_clients"]
                    ),
                )
            return self._federated

    def serve_forever(self) -> None:
        listener = self._create_listener()
        address = listener.getsockname()
        monitor = ResourceMonitor(self.config)
        monitor.start()

        self.logger.write(
            "server_start",
            listen_host=address[0],
            listen_port=address[1],
            transport=self.config["transport"]["kind"],
            task=self.config["execution"]["task"],
            deployment=self.config["execution"]["deployment"],
            resource_monitor_enabled=monitor.enabled,
            resource_interval_ms=monitor.interval_ms,
        )
        print(
            f"[server] listening on {address[0]}:{address[1]} "
            f"using {self.config['transport']['kind']} "
            f"task={self.config['execution']['task']} "
            f"deployment={self.config['execution']['deployment']}"
        )
        if (
            self.config["execution"]["task"] == "training"
            and self.config["execution"]["deployment"] == "federated"
        ):
            print(
                f"[server] synchronous FL: "
                f"rounds={self.config['federated']['rounds']} "
                f"expected_clients="
                f"{self.config['federated']['expected_clients']} "
                f"aggregation={self.config['federated']['aggregation']}"
            )

        try:
            while not self.stop_event.is_set():
                conn, peer = listener.accept()
                thread = threading.Thread(
                    target=self._handle_connection,
                    args=(conn, peer),
                    daemon=True,
                )
                thread.start()
        finally:
            listener.close()
            summary = monitor.stop()
            self.logger.write(
                "resource_summary",
                resource_csv=str(monitor.csv_path),
                resource_summary_json=str(
                    monitor.summary_path
                ),
                bytes_sent=summary.get("bytes_sent"),
                bytes_received=summary.get(
                    "bytes_received"
                ),
                cpu_energy_j=summary.get("cpu_energy_j"),
                gpu_energy_j=summary.get("gpu_energy_j"),
                system_energy_j=summary.get(
                    "system_energy_j"
                ),
            )
            self.logger.write("server_stop")

    def _handle_connection(
        self,
        raw_conn: socket.socket,
        peer,
    ) -> None:
        conn = raw_conn
        try:
            conn = self._wrap_tls(raw_conn)
            self.logger.write(
                "client_connected",
                peer=str(peer),
            )

            while True:
                header, payload = recv_frame(conn)
                op = header.get("op")
                request_id = header.get("request_id")

                if op == "infer":
                    self._handle_infer(
                        conn,
                        request_id,
                        payload,
                    )

                elif op == "train_batch":
                    self._handle_train_batch(
                        conn,
                        request_id,
                        header,
                        payload,
                    )

                elif op == "fl_get":
                    self._handle_fl_get(
                        conn,
                        request_id,
                        header,
                    )

                elif op == "fl_update":
                    self._handle_fl_update(
                        conn,
                        request_id,
                        header,
                        payload,
                    )

                elif op == "model_download":
                    self._handle_model_download(
                        conn,
                        request_id,
                    )

                elif op == "close":
                    send_frame(
                        conn,
                        {
                            "status": "ok",
                            "request_id": request_id,
                        },
                    )
                    break

                else:
                    raise RuntimeError(
                        f"Unsupported operation: {op!r}"
                    )

        except (ConnectionError, OSError):
            pass
        except Exception as exc:
            self.logger.write(
                "server_error",
                peer=str(peer),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            try:
                send_frame(
                    conn,
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
            traceback.print_exc()
        finally:
            try:
                conn.close()
            except Exception:
                pass
            self.logger.write(
                "client_disconnected",
                peer=str(peer),
            )

    def _handle_infer(
        self,
        conn: socket.socket,
        request_id: str,
        payload: bytes,
    ) -> None:
        workload = self._ensure_workload()
        array = bytes_to_array(payload)

        start_ns = time.perf_counter_ns()
        with self._workload_lock:
            output = workload.infer(array)
        end_ns = time.perf_counter_ns()

        response_payload = array_to_bytes(
            np.asarray(output)
        )
        send_frame(
            conn,
            {
                "status": "ok",
                "request_id": request_id,
            },
            response_payload,
        )
        self.logger.write(
            "inference_completed",
            request_id=request_id,
            input_bytes=len(payload),
            output_bytes=len(response_payload),
            inference_ms=(
                end_ns - start_ns
            ) / 1_000_000.0,
        )

    def _handle_train_batch(
        self,
        conn: socket.socket,
        request_id: str,
        header: Dict[str, Any],
        payload: bytes,
    ) -> None:
        workload = self._ensure_workload()
        inputs, targets = bytes_to_training_batch(payload)

        start_ns = time.perf_counter_ns()
        with self._workload_lock:
            metrics = workload.train_batch(
                inputs,
                targets,
            )
        end_ns = time.perf_counter_ns()

        train_step_ms = (
            end_ns - start_ns
        ) / 1_000_000.0

        send_frame(
            conn,
            {
                "status": "ok",
                "request_id": request_id,
                "loss": metrics.get("loss"),
                "accuracy": metrics.get("accuracy"),
                "learning_rate": metrics.get(
                    "learning_rate"
                ),
                "train_step_ms": train_step_ms,
            },
        )

        self.logger.write(
            "training_completed",
            request_id=request_id,
            epoch=header.get("epoch"),
            batch_index=header.get("batch_index"),
            input_bytes=len(payload),
            samples_processed=int(inputs.shape[0]),
            loss=metrics.get("loss"),
            accuracy=metrics.get("accuracy"),
            learning_rate=metrics.get(
                "learning_rate"
            ),
            train_step_ms=train_step_ms,
        )

    def _handle_fl_get(
        self,
        conn: socket.socket,
        request_id: str,
        header: Dict[str, Any],
    ) -> None:
        coordinator = self._ensure_federated_coordinator()
        round_index, parameters, done = (
            coordinator.get_global()
        )

        payload = (
            b""
            if done
            else arrays_to_bytes(parameters)
        )

        send_frame(
            conn,
            {
                "status": "ok",
                "request_id": request_id,
                "round": round_index,
                "done": done,
            },
            payload,
        )

        self.logger.write(
            "federated_phase",
            round=round_index,
            client_id=header.get("client_id"),
            phase="Download",
            bytes_sent=len(payload),
            done=done,
        )

    def _handle_fl_update(
        self,
        conn: socket.socket,
        request_id: str,
        header: Dict[str, Any],
        payload: bytes,
    ) -> None:
        coordinator = self._ensure_federated_coordinator()
        round_index = int(header["round"])
        client_id = str(header["client_id"])
        parameters = bytes_to_arrays(payload)

        self.logger.write(
            "federated_phase",
            round=round_index,
            client_id=client_id,
            phase="Upload",
            bytes_received=len(payload),
            num_examples=int(
                header.get("num_examples", 1)
            ),
            client_loss=header.get("loss"),
            client_accuracy=header.get("accuracy"),
        )

        next_round, done = coordinator.submit_update(
            round_index=round_index,
            client_id=client_id,
            parameters=parameters,
            num_examples=int(
                header.get("num_examples", 1)
            ),
            metrics={
                "loss": header.get("loss"),
                "accuracy": header.get("accuracy"),
            },
        )

        send_frame(
            conn,
            {
                "status": "ok",
                "request_id": request_id,
                "next_round": next_round,
                "done": done,
            },
        )

        self.logger.write(
            "federated_round_progress",
            completed_round=round_index,
            next_round=next_round,
            client_id=client_id,
            done=done,
        )

    def _handle_model_download(
        self,
        conn: socket.socket,
        request_id: str,
    ) -> None:
        artifact = self.config["ai"].get(
            "model_artifact"
        )
        if not artifact:
            raise RuntimeError(
                "model_download requires "
                "ai.model_artifact on the server"
            )

        data = Path(artifact).read_bytes()
        send_frame(
            conn,
            {
                "status": "ok",
                "request_id": request_id,
            },
            data,
        )
        self.logger.write(
            "model_download_completed",
            request_id=request_id,
            bytes_sent=len(data),
        )
