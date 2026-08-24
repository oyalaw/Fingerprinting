from __future__ import annotations

import socket
import ssl
import time
import uuid
from pathlib import Path
from statistics import mean
from typing import Any, Dict

import numpy as np

from .datasets import InputGenerator
from .metadata import EventLogger
from .protocol import (
    array_to_bytes,
    arrays_to_bytes,
    bytes_to_array,
    bytes_to_arrays,
    recv_frame,
    send_frame,
    training_batch_to_bytes,
)
from .resource_monitor import ResourceMonitor
from .workloads import build_workload


class ExperimentClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = EventLogger(config)
        self.generator = InputGenerator(config)

    def _connect(self) -> socket.socket:
        host = self.config["node"]["host"]
        port = int(self.config["node"]["port"])
        sock = socket.create_connection((host, port), timeout=30)

        transport = self.config["transport"]
        if transport["kind"] != "tls":
            return sock

        verify_peer = bool(transport.get("verify_peer"))
        if verify_peer:
            context = ssl.create_default_context(
                cafile=transport.get("cafile")
            )
        else:
            context = ssl._create_unverified_context()

        server_hostname = transport.get("server_hostname")
        if not server_hostname:
            server_hostname = host if verify_peer else None

        return context.wrap_socket(
            sock,
            server_hostname=server_hostname,
        )

    def run(self) -> None:
        execution = self.config["execution"]
        task = execution["task"]
        deployment = execution["deployment"]
        operation = execution.get("operation", "workload")

        monitor = ResourceMonitor(self.config)
        self.logger.write(
            "experiment_start",
            task=task,
            deployment=deployment,
            resource_monitor_enabled=monitor.enabled,
            resource_interval_ms=monitor.interval_ms,
        )
        monitor.start()

        try:
            if operation == "model_download":
                self._run_model_download()
            elif task == "inference":
                if deployment == "local":
                    self._run_local_inference()
                elif deployment == "remote":
                    self._run_remote_inference()
                else:
                    raise ValueError(
                        f"Unsupported inference deployment: {deployment}"
                    )
            elif task == "training":
                if deployment == "local":
                    self._run_local_training()
                elif deployment == "remote":
                    self._run_remote_training()
                elif deployment == "federated":
                    self._run_federated_training()
                else:
                    raise ValueError(
                        f"Unsupported training deployment: {deployment}"
                    )
            else:
                raise ValueError(f"Unsupported task: {task}")
        finally:
            summary = monitor.stop()
            self.logger.write(
                "resource_summary",
                resource_csv=str(monitor.csv_path),
                resource_summary_json=str(monitor.summary_path),
                bytes_sent=summary.get("bytes_sent"),
                bytes_received=summary.get("bytes_received"),
                cpu_energy_j=summary.get("cpu_energy_j"),
                gpu_energy_j=summary.get("gpu_energy_j"),
                system_energy_j=summary.get("system_energy_j"),
            )
            self.logger.write("experiment_stop")

    def _run_remote_inference(self) -> None:
        repetitions = int(self.config["execution"]["repetitions"])
        warmup = int(self.config["execution"]["warmup"])
        interval = (
            float(self.config["execution"]["interval_ms"]) / 1000.0
        )

        with self._connect() as sock:
            for index in range(warmup + repetitions):
                request_id = uuid.uuid4().hex
                array = self.generator.sample()
                payload = array_to_bytes(array)

                start_ns = time.perf_counter_ns()
                send_frame(
                    sock,
                    {
                        "op": "infer",
                        "request_id": request_id,
                    },
                    payload,
                )
                header, response_payload = recv_frame(sock)
                end_ns = time.perf_counter_ns()

                if header.get("status") != "ok":
                    raise RuntimeError(
                        header.get(
                            "error",
                            "remote inference failed",
                        )
                    )

                output = bytes_to_array(response_payload)

                if index >= warmup:
                    self.logger.write(
                        "inference_request",
                        request_id=request_id,
                        sequence_index=index - warmup,
                        request_bytes=len(payload),
                        response_bytes=len(response_payload),
                        round_trip_ms=(
                            end_ns - start_ns
                        ) / 1_000_000.0,
                        output_shape=list(output.shape),
                    )

                if interval > 0:
                    time.sleep(interval)

            self._close_remote(sock)

    def _run_local_inference(self) -> None:
        repetitions = int(self.config["execution"]["repetitions"])
        warmup = int(self.config["execution"]["warmup"])
        workload = build_workload(self.config)

        for index in range(warmup + repetitions):
            array = self.generator.sample()
            start_ns = time.perf_counter_ns()
            output = workload.infer(array)
            end_ns = time.perf_counter_ns()

            if index >= warmup:
                self.logger.write(
                    "local_inference",
                    sequence_index=index - warmup,
                    inference_ms=(
                        end_ns - start_ns
                    ) / 1_000_000.0,
                    output_shape=list(output.shape),
                )

    def _run_local_training(self) -> None:
        execution = self.config["execution"]
        epochs = int(execution["epochs"])
        steps = int(execution["steps_per_epoch"])
        workload = build_workload(self.config)

        global_step = 0
        for epoch in range(epochs):
            epoch_start = time.perf_counter_ns()
            losses = []
            accuracies = []

            for batch_index in range(steps):
                inputs, targets = self.generator.training_batch()
                start_ns = time.perf_counter_ns()
                metrics = workload.train_batch(inputs, targets)
                end_ns = time.perf_counter_ns()

                losses.append(float(metrics["loss"]))
                if "accuracy" in metrics:
                    accuracies.append(float(metrics["accuracy"]))

                self.logger.write(
                    "training_step",
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=global_step,
                    samples_processed=int(inputs.shape[0]),
                    loss=float(metrics["loss"]),
                    accuracy=metrics.get("accuracy"),
                    learning_rate=metrics.get("learning_rate"),
                    train_step_ms=(
                        end_ns - start_ns
                    ) / 1_000_000.0,
                )
                global_step += 1

            epoch_end = time.perf_counter_ns()
            self.logger.write(
                "training_epoch",
                epoch=epoch,
                steps=steps,
                loss_mean=mean(losses) if losses else None,
                accuracy_mean=(
                    mean(accuracies)
                    if accuracies
                    else None
                ),
                epoch_time_ms=(
                    epoch_end - epoch_start
                ) / 1_000_000.0,
            )

    def _run_remote_training(self) -> None:
        execution = self.config["execution"]
        epochs = int(execution["epochs"])
        steps = int(execution["steps_per_epoch"])

        with self._connect() as sock:
            global_step = 0
            for epoch in range(epochs):
                losses = []
                accuracies = []
                epoch_start = time.perf_counter_ns()

                for batch_index in range(steps):
                    request_id = uuid.uuid4().hex
                    inputs, targets = self.generator.training_batch()
                    payload = training_batch_to_bytes(
                        inputs,
                        targets,
                    )

                    start_ns = time.perf_counter_ns()
                    send_frame(
                        sock,
                        {
                            "op": "train_batch",
                            "request_id": request_id,
                            "epoch": epoch,
                            "batch_index": batch_index,
                        },
                        payload,
                    )
                    header, response_payload = recv_frame(sock)
                    end_ns = time.perf_counter_ns()

                    if header.get("status") != "ok":
                        raise RuntimeError(
                            header.get(
                                "error",
                                "remote training failed",
                            )
                        )

                    loss = float(header["loss"])
                    accuracy = header.get("accuracy")
                    losses.append(loss)
                    if accuracy is not None:
                        accuracies.append(float(accuracy))

                    self.logger.write(
                        "remote_training_step",
                        request_id=request_id,
                        epoch=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                        samples_processed=int(inputs.shape[0]),
                        request_bytes=len(payload),
                        response_bytes=len(response_payload),
                        loss=loss,
                        accuracy=accuracy,
                        learning_rate=header.get("learning_rate"),
                        server_train_step_ms=header.get(
                            "train_step_ms"
                        ),
                        round_trip_ms=(
                            end_ns - start_ns
                        ) / 1_000_000.0,
                    )
                    global_step += 1

                epoch_end = time.perf_counter_ns()
                self.logger.write(
                    "remote_training_epoch",
                    epoch=epoch,
                    steps=steps,
                    loss_mean=mean(losses) if losses else None,
                    accuracy_mean=(
                        mean(accuracies)
                        if accuracies
                        else None
                    ),
                    epoch_time_ms=(
                        epoch_end - epoch_start
                    ) / 1_000_000.0,
                )

            self._close_remote(sock)

    def _run_federated_training(self) -> None:
        fed = self.config["federated"]
        configured_rounds = int(fed["rounds"])
        local_epochs = int(fed["local_epochs"])
        steps = int(fed["steps_per_epoch"])
        client_id = str(fed["client_id"])
        workload = build_workload(self.config)

        with self._connect() as sock:
            for _ in range(configured_rounds):
                request_id = uuid.uuid4().hex
                download_start = time.perf_counter_ns()
                send_frame(
                    sock,
                    {
                        "op": "fl_get",
                        "request_id": request_id,
                        "client_id": client_id,
                    },
                )
                header, payload = recv_frame(sock)
                download_end = time.perf_counter_ns()

                if header.get("status") != "ok":
                    raise RuntimeError(
                        header.get(
                            "error",
                            "federated download failed",
                        )
                    )

                if header.get("done"):
                    break

                round_index = int(header["round"])
                parameters = bytes_to_arrays(payload)
                workload.set_parameters(parameters)

                self.logger.write(
                    "federated_phase",
                    round=round_index,
                    client_id=client_id,
                    phase="Download",
                    bytes_received=len(payload),
                    phase_time_ms=(
                        download_end - download_start
                    ) / 1_000_000.0,
                )

                train_start = time.perf_counter_ns()
                losses = []
                accuracies = []
                examples = 0

                for local_epoch in range(local_epochs):
                    for batch_index in range(steps):
                        inputs, targets = self.generator.training_batch()
                        metrics = workload.train_batch(
                            inputs,
                            targets,
                        )
                        examples += int(inputs.shape[0])
                        losses.append(float(metrics["loss"]))
                        if "accuracy" in metrics:
                            accuracies.append(
                                float(metrics["accuracy"])
                            )

                        self.logger.write(
                            "federated_local_step",
                            round=round_index,
                            client_id=client_id,
                            local_epoch=local_epoch,
                            batch_index=batch_index,
                            samples_processed=int(inputs.shape[0]),
                            loss=float(metrics["loss"]),
                            accuracy=metrics.get("accuracy"),
                            learning_rate=metrics.get(
                                "learning_rate"
                            ),
                        )

                train_end = time.perf_counter_ns()
                self.logger.write(
                    "federated_phase",
                    round=round_index,
                    client_id=client_id,
                    phase="Training",
                    local_epochs=local_epochs,
                    steps_per_epoch=steps,
                    samples_processed=examples,
                    loss_mean=mean(losses) if losses else None,
                    accuracy_mean=(
                        mean(accuracies)
                        if accuracies
                        else None
                    ),
                    phase_time_ms=(
                        train_end - train_start
                    ) / 1_000_000.0,
                )

                update_payload = arrays_to_bytes(
                    workload.get_parameters()
                )
                upload_start = time.perf_counter_ns()
                send_frame(
                    sock,
                    {
                        "op": "fl_update",
                        "request_id": uuid.uuid4().hex,
                        "client_id": client_id,
                        "round": round_index,
                        "num_examples": examples,
                        "loss": mean(losses) if losses else None,
                        "accuracy": (
                            mean(accuracies)
                            if accuracies
                            else None
                        ),
                    },
                    update_payload,
                )
                ack, ack_payload = recv_frame(sock)
                upload_end = time.perf_counter_ns()

                if ack.get("status") != "ok":
                    raise RuntimeError(
                        ack.get(
                            "error",
                            "federated upload failed",
                        )
                    )

                self.logger.write(
                    "federated_phase",
                    round=round_index,
                    client_id=client_id,
                    phase="Upload",
                    bytes_sent=len(update_payload),
                    next_round=ack.get("next_round"),
                    done=ack.get("done", False),
                    phase_time_ms=(
                        upload_end - upload_start
                    ) / 1_000_000.0,
                )

                if ack.get("done"):
                    break

            self._close_remote(sock)

    def _run_model_download(self) -> None:
        request_id = uuid.uuid4().hex
        with self._connect() as sock:
            start_ns = time.perf_counter_ns()
            send_frame(
                sock,
                {
                    "op": "model_download",
                    "request_id": request_id,
                },
            )
            header, payload = recv_frame(sock)
            end_ns = time.perf_counter_ns()

            if header.get("status") != "ok":
                raise RuntimeError(
                    header.get(
                        "error",
                        "model download failed",
                    )
                )

            output_dir = Path(
                self.config["experiment"]["output_dir"]
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / (
                f"{self.config['experiment']['experiment_id']}"
                "_downloaded_model.bin"
            )
            target.write_bytes(payload)

            self.logger.write(
                "model_download",
                request_id=request_id,
                bytes_received=len(payload),
                round_trip_ms=(
                    end_ns - start_ns
                ) / 1_000_000.0,
                saved_to=str(target),
            )

            self._close_remote(sock)

    def _close_remote(self, sock: socket.socket) -> None:
        send_frame(
            sock,
            {
                "op": "close",
                "request_id": uuid.uuid4().hex,
            },
        )
        recv_frame(sock)
