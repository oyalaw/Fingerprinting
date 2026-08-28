from __future__ import annotations

import copy
import json
import socket
import ssl
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .federated import SynchronousFedAvgCoordinator
from .datasets import InputGenerator
from .federated_contract import (
    build_model_contract,
    compact_contract,
    write_model_contract,
)
from .federated_policy import build_training_policy
from .experiment_coordination import ActiveRunCoordinator, ensure_run_id
from .experiment_layout import materialize_role_metadata
from .experiment_integrity import RoundCheckpointManager, disk_preflight
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
from .tls import ensure_server_tls_material
from .training_metrics import (
    PerformanceLogWriter,
    collect_anomaly_batches,
    evaluate_anomaly_batches,
    evaluate_generator,
    mean_std_update_norms,
    metric_delta,
    parameter_delta_l2_norm,
    parameter_l2_norm,
    utc_now_iso,
)
from .workloads import build_workload


class ExperimentServer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._tls_material = ensure_server_tls_material(
            self.config
        )
        self.logger = EventLogger(self.config)
        self.workload = None
        self.stop_event = threading.Event()
        ensure_run_id(self.config)
        # runner materializes metadata before the server creates its neutral
        # run_id. Refresh now so server manifests do not retain run_id=null.
        materialize_role_metadata(self.config)
        self._active_run_coordinator = None

        self._workload_lock = threading.Lock()
        self._federated_lock = threading.Lock()
        self._federated = None
        self._model_contract = None

        self._performance_cfg = self.config.get("performance_logging", {})
        self._performance_enabled = bool(
            self._performance_cfg.get("enabled", True)
        )
        self._performance_writer = (
            PerformanceLogWriter(self.config)
            if self._performance_enabled
            else None
        )
        self._performance_eval_generator = None
        self._performance_eval_error: str | None = None
        self._anomaly_calibration_data = None
        self._anomaly_evaluation_data = None
        self._performance_lock = threading.Lock()
        self._round_download_bytes: Dict[int, int] = {}
        self._round_started_perf_ns: Dict[int, int] = {}
        self._round_started_utc: Dict[int, str] = {}
        self._previous_global_metrics: Dict[str, Any] = {}
        self._ready_condition = threading.Condition()
        self._ready_clients: set[str] = set()
        self._partition_slots: Dict[str, int] = {}
        self._final_ack_clients: set[str] = set()
        self._checkpoint_manager = RoundCheckpointManager(self.config)
        self._experiment_completed = False
        self._experiment_terminal = False

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
        minimum = str(
            transport.get("minimum_tls_version", "TLSv1_2")
        )
        context.minimum_version = (
            ssl.TLSVersion.TLSv1_3
            if minimum == "TLSv1_3"
            else ssl.TLSVersion.TLSv1_2
        )
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
                    on_round_aggregated=self._on_round_aggregated,
                )
            return self._federated

    def _ensure_performance_eval_generator(self):
        if not self._performance_enabled:
            return None
        if self._performance_eval_generator is not None:
            return self._performance_eval_generator
        if self._performance_eval_error is not None:
            return None

        eval_config = copy.deepcopy(self.config)
        eval_config["data"]["split"] = str(
            self._performance_cfg.get("server_eval_split", "test")
        )
        eval_config["data"]["shuffle"] = False
        eval_config["execution"]["seed"] = int(
            self.config["execution"].get("seed", 42)
        ) + 100_000
        try:
            self._performance_eval_generator = InputGenerator(eval_config)
        except Exception as exc:
            self._performance_eval_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.logger.write(
                "server_performance_evaluation_setup_error",
                error_type=type(exc).__name__,
                error=str(exc),
                evaluation_split=eval_config["data"]["split"],
            )
            if bool(
                self._performance_cfg.get(
                    "server_evaluation_required", False
                )
            ):
                raise
        return self._performance_eval_generator

    def _ensure_anomaly_evaluation_data(self):
        if not self._performance_enabled:
            return None, None
        if (
            self._anomaly_calibration_data is not None
            and self._anomaly_evaluation_data is not None
        ):
            return self._anomaly_calibration_data, self._anomaly_evaluation_data
        if self._performance_eval_error is not None:
            return None, None

        anomaly_cfg = self.config.get("anomaly_detection", {}) or {}
        eval_batch_size = int(anomaly_cfg.get("evaluation_batch_size", 32))
        try:
            calibration_config = copy.deepcopy(self.config)
            calibration_config["data"]["split"] = "train"
            calibration_config.setdefault("anomaly_detection", {})["data_role"] = "calibration"
            calibration_config["data"]["shuffle"] = False
            calibration_config["execution"]["batch_size"] = eval_batch_size
            evaluation_config = copy.deepcopy(self.config)
            evaluation_config["data"]["split"] = str(
                self._performance_cfg.get("server_eval_split", "test")
            )
            evaluation_config["data"]["shuffle"] = False
            evaluation_config["execution"]["batch_size"] = eval_batch_size
            calibration_generator = InputGenerator(calibration_config)
            evaluation_generator = InputGenerator(evaluation_config)
            self._anomaly_calibration_data = collect_anomaly_batches(
                calibration_generator,
                int(anomaly_cfg.get("calibration_batches", 10)),
                normal_only=True,
            )
            self._anomaly_evaluation_data = collect_anomaly_batches(
                evaluation_generator,
                int(anomaly_cfg.get("evaluation_batches", 10)),
                normal_only=False,
            )
            self.logger.write(
                "server_anomaly_evaluation_prepared",
                anomaly_labels=anomaly_cfg.get("anomaly_labels"),
                threshold_percentile=anomaly_cfg.get("threshold_percentile"),
                calibration_batches=len(self._anomaly_calibration_data),
                evaluation_batches=len(self._anomaly_evaluation_data),
                evaluation_batch_size=eval_batch_size,
            )
        except Exception as exc:
            self._performance_eval_error = f"{type(exc).__name__}: {exc}"
            self.logger.write(
                "server_performance_evaluation_setup_error",
                error_type=type(exc).__name__,
                error=str(exc),
                evaluation_split=self._performance_cfg.get("server_eval_split", "test"),
            )
            if bool(self._performance_cfg.get("server_evaluation_required", False)):
                raise
        return self._anomaly_calibration_data, self._anomaly_evaluation_data

    def _record_round_download(self, round_index: int, payload_bytes: int) -> None:
        if not self._performance_enabled:
            return
        with self._performance_lock:
            if round_index not in self._round_started_perf_ns:
                self._round_started_perf_ns[round_index] = time.perf_counter_ns()
                self._round_started_utc[round_index] = utc_now_iso()
            self._round_download_bytes[round_index] = (
                self._round_download_bytes.get(round_index, 0)
                + int(payload_bytes)
            )

    def _on_round_aggregated(
        self,
        round_index: int,
        updates,
        previous_global,
        new_global,
        aggregation_time_ms: float,
    ) -> None:
        if not self._performance_enabled or self._performance_writer is None:
            self._checkpoint_manager.record_round(
                round_index, new_global, len(updates)
            )
            return

        evaluation_metrics: Dict[str, Any] = {}
        evaluation_time_ms = 0.0
        evaluation_error = self._performance_eval_error
        if str(self.config.get("ai", {}).get("application", "")) == "anomaly_detection":
            calibration_data, evaluation_data = self._ensure_anomaly_evaluation_data()
            if calibration_data is not None and evaluation_data is not None:
                try:
                    evaluation_metrics, evaluation_time_ms = evaluate_anomaly_batches(
                        self._ensure_workload(),
                        calibration_data,
                        evaluation_data,
                        threshold_percentile=float(
                            self.config.get("anomaly_detection", {}).get(
                                "threshold_percentile", 95.0
                            )
                        ),
                    )
                except Exception as exc:
                    evaluation_error = f"{type(exc).__name__}: {exc}"
                    self.logger.write(
                        "server_performance_evaluation_error",
                        round=round_index,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    if bool(self._performance_cfg.get("server_evaluation_required", False)):
                        raise
        else:
            generator = self._ensure_performance_eval_generator()
            if generator is not None:
                try:
                    generator.reset()
                    evaluation_metrics, evaluation_time_ms = evaluate_generator(
                        self._ensure_workload(),
                        generator,
                        int(self._performance_cfg.get("server_eval_batches", 10)),
                    )
                except Exception as exc:
                    evaluation_error = f"{type(exc).__name__}: {exc}"
                    self.logger.write(
                        "server_performance_evaluation_error",
                        round=round_index,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    if bool(
                        self._performance_cfg.get(
                            "server_evaluation_required", False
                        )
                    ):
                        raise

        update_norm_mean, update_norm_std = mean_std_update_norms(
            list(updates.values())
        )
        new_global_norm = parameter_l2_norm(new_global)
        global_update_norm = parameter_delta_l2_norm(
            new_global, previous_global
        )
        model_size_bytes = len(arrays_to_bytes(new_global))

        with self._performance_lock:
            bytes_sent_round = self._round_download_bytes.pop(
                round_index, 0
            )
            started_ns = self._round_started_perf_ns.pop(
                round_index, None
            )
            started_utc = self._round_started_utc.pop(
                round_index, utc_now_iso()
            )
        round_duration_sec = (
            (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
            if started_ns is not None
            else None
        )

        total_examples = sum(
            int(update.num_examples) for update in updates.values()
        )
        bytes_received_round = sum(
            int(update.metrics.get("payload_bytes", 0) or 0)
            for update in updates.values()
        )

        row = {
            "experiment_id": self.config["experiment"]["experiment_id"],
            "round": round_index,
            "family": self.config["ai"]["family"],
            "architecture": self.config["ai"]["architecture"],
            "variant": self.config["ai"]["variant"],
            "application": self.config["ai"]["application"],
            "dataset": self.config["ai"]["dataset"],
            "framework": self.config["ai"]["framework"],
            "runtime": self.config["ai"]["runtime"],
            "aggregation_rule": self.config["federated"].get(
                "aggregation", "fedavg"
            ),
            "input_size": self.config["ai"]["input_size"],
            "batch_size": self.config["execution"]["batch_size"],
            "learning_rate": self.config["execution"]["learning_rate"],
            "global_rounds": self.config["federated"]["rounds"],
            "local_epochs": self.config["federated"]["local_epochs"],
            "steps_per_epoch": self.config["federated"]["steps_per_epoch"],
            "training_policy_id": build_training_policy(self.config).get("policy_id"),
            "policy_source": "server",
            "partition_type": self.config.get("data", {}).get("partition", {}).get("type"),
            "partition_alpha": self.config.get("data", {}).get("partition", {}).get("alpha"),
            "partition_seed": self.config.get("data", {}).get("partition", {}).get("seed"),
            "partition_disjoint": self.config.get("data", {}).get("partition", {}).get("disjoint", True),
            "clients_expected": int(
                self.config["federated"]["expected_clients"]
            ),
            "clients_received": len(updates),
            "client_ids": ";".join(sorted(updates)),
            "total_examples": total_examples,
            "aggregation_time_ms": aggregation_time_ms,
            "evaluation_time_ms": evaluation_time_ms,
            "global_loss": evaluation_metrics.get("loss"),
            "global_accuracy": evaluation_metrics.get("accuracy"),
            "global_precision": evaluation_metrics.get("precision"),
            "global_recall": evaluation_metrics.get("recall"),
            "global_f1": evaluation_metrics.get("f1"),
            "global_macro_precision": evaluation_metrics.get("macro_precision"),
            "global_macro_recall": evaluation_metrics.get("macro_recall"),
            "global_macro_f1": evaluation_metrics.get("macro_f1"),
            "global_reconstruction_loss": evaluation_metrics.get(
                "reconstruction_loss"
            ),
            "global_mse": evaluation_metrics.get("mse"),
            "global_mae": evaluation_metrics.get("mae"),
            "global_kl_loss": evaluation_metrics.get("kl_loss"),
            "vae_beta": evaluation_metrics.get("vae_beta"),
            "evaluation_samples": evaluation_metrics.get(
                "evaluated_samples"
            ),
            "anomaly_accuracy": evaluation_metrics.get("anomaly_accuracy"),
            "anomaly_precision": evaluation_metrics.get("anomaly_precision"),
            "anomaly_recall": evaluation_metrics.get("anomaly_recall"),
            "anomaly_f1": evaluation_metrics.get("anomaly_f1"),
            "anomaly_auroc": evaluation_metrics.get("anomaly_auroc"),
            "anomaly_auprc": evaluation_metrics.get("anomaly_auprc"),
            "anomaly_threshold": evaluation_metrics.get("anomaly_threshold"),
            "anomaly_threshold_percentile": evaluation_metrics.get("anomaly_threshold_percentile"),
            "anomaly_tp": evaluation_metrics.get("anomaly_tp"),
            "anomaly_fp": evaluation_metrics.get("anomaly_fp"),
            "anomaly_tn": evaluation_metrics.get("anomaly_tn"),
            "anomaly_fn": evaluation_metrics.get("anomaly_fn"),
            "anomaly_eval_samples": evaluation_metrics.get("anomaly_eval_samples"),
            "anomaly_samples": evaluation_metrics.get("anomaly_samples"),
            "normal_samples": evaluation_metrics.get("normal_samples"),
            "normal_error_mean": evaluation_metrics.get("normal_error_mean"),
            "anomaly_error_mean": evaluation_metrics.get("anomaly_error_mean"),
            "evaluation_split": self._performance_cfg.get(
                "server_eval_split", "test"
            ),
            "global_model_norm_l2": new_global_norm,
            "global_update_norm_l2": global_update_norm,
            "mean_client_update_norm_l2": update_norm_mean,
            "std_client_update_norm_l2": update_norm_std,
            "model_size_bytes": model_size_bytes,
            "bytes_received_round": bytes_received_round,
            "bytes_sent_round": bytes_sent_round,
            "round_duration_sec": round_duration_sec,
            "loss_change": metric_delta(
                evaluation_metrics.get("loss"),
                self._previous_global_metrics.get("loss"),
                improvement_direction="down",
            ),
            "accuracy_change": metric_delta(
                evaluation_metrics.get("accuracy"),
                self._previous_global_metrics.get("accuracy"),
            ),
            "f1_change": metric_delta(
                evaluation_metrics.get("f1"),
                self._previous_global_metrics.get("f1"),
            ),
            "timestamp_start_utc": started_utc,
            "timestamp_end_utc": utc_now_iso(),
            "evaluation_error": evaluation_error,
        }
        self._performance_writer.write_server_round(row)
        self._previous_global_metrics = dict(evaluation_metrics)
        self.logger.write(
            "federated_global_metrics",
            round=round_index,
            metrics_csv=str(
                self._performance_writer.output_dir / "round_metrics.csv"
            ),
            global_loss=row.get("global_loss"),
            global_accuracy=row.get("global_accuracy"),
            global_precision=row.get("global_precision"),
            global_recall=row.get("global_recall"),
            global_f1=row.get("global_f1"),
            global_update_norm_l2=global_update_norm,
            evaluation_time_ms=evaluation_time_ms,
            evaluation_error=evaluation_error,
        )
        self._checkpoint_manager.record_round(
            round_index, new_global, len(updates)
        )

    def serve_forever(self) -> None:
        listener = self._create_listener()
        listener.settimeout(1.0)
        address = listener.getsockname()
        coordination_cfg = self.config.get("coordination", {}) or {}
        if bool(coordination_cfg.get("enabled", True)):
            self._active_run_coordinator = ActiveRunCoordinator(self.config)
            self._active_run_coordinator.start()
            print(
                "[server] experiment coordinator: "
                f"{self._active_run_coordinator.host}:"
                f"{self._active_run_coordinator.port} "
                f"run_id={self._active_run_coordinator.run_id}"
            )
        if (
            self.config.get("execution", {}).get("task") == "training"
            and self.config.get("execution", {}).get("deployment") == "federated"
        ):
            checkpoint_cfg = self.config.get("checkpoint", {}) or {}
            disk_check = disk_preflight(
                self.config["experiment"]["output_dir"],
                float(checkpoint_cfg.get("minimum_free_disk_gb", 5.0)),
            )
            if not disk_check.get("passed", False):
                raise RuntimeError(
                    "Insufficient server disk space for checkpoints: "
                    f"{disk_check.get('free_gb', 0.0):.2f} GiB available, "
                    f"{disk_check.get('minimum_free_gb', 0.0):.2f} GiB required"
                )
            self._checkpoint_manager.start()

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
            if self._performance_enabled:
                if str(self.config.get("ai", {}).get("application", "")) == "anomaly_detection":
                    self._ensure_anomaly_evaluation_data()
                    anomaly_cfg = self.config.get("anomaly_detection", {}) or {}
                    print(
                        "[server] anomaly performance logging enabled: "
                        f"labels={anomaly_cfg.get('anomaly_labels')} "
                        f"threshold_percentile={anomaly_cfg.get('threshold_percentile', 95.0)} "
                        f"calibration_batches={anomaly_cfg.get('calibration_batches', 10)} "
                        f"evaluation_batches={anomaly_cfg.get('evaluation_batches', 10)} "
                        f"evaluation_batch_size={anomaly_cfg.get('evaluation_batch_size', 32)}"
                    )
                else:
                    self._ensure_performance_eval_generator()
                    print(
                        "[server] per-round performance logging enabled: "
                        f"evaluation_split={self._performance_cfg.get('server_eval_split', 'test')} "
                        f"evaluation_batches={self._performance_cfg.get('server_eval_batches', 10)}"
                    )
            partition = self.config.get("data", {}).get("partition", {}) or {}
            partition_policy_path = (
                Path(self.config["experiment"]["output_dir"])
                / "data_partition_policy.json"
            )
            partition_policy_path.write_text(
                json.dumps(
                    {
                        "type": partition.get("type", "iid"),
                        "alpha": partition.get("alpha"),
                        "seed": partition.get("seed"),
                        "client_count": int(
                            partition.get("client_count")
                            or self.config["federated"]["expected_clients"]
                        ),
                        "disjoint": bool(partition.get("disjoint", True)),
                        "source": "server",
                        "anomaly_labels": (
                            self.config.get("anomaly_detection", {}).get("anomaly_labels")
                            if self.config.get("ai", {}).get("application") == "anomaly_detection"
                            else None
                        ),
                        "anomaly_training_excludes_labels": (
                            self.config.get("ai", {}).get("application") == "anomaly_detection"
                        ),
                        "note": (
                            "IID and non-IID partitions are disjoint. For non-IID, "
                            "class labels shape input allocation only; autoencoder "
                            "reconstruction targets remain x->x."
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            print(
                "[server] data partition: "
                f"type={partition.get('type', 'iid')} "
                + (
                    f"alpha={partition.get('alpha')} "
                    if partition.get("type") == "non_iid"
                    else ""
                )
                + f"seed={partition.get('seed')} disjoint=true"
            )
            print(
                f"[server] synchronous FL: "
                f"rounds={self.config['federated']['rounds']} "
                f"expected_clients="
                f"{self.config['federated']['expected_clients']} "
                f"aggregation={self.config['federated']['aggregation']}"
            )

        try:
            while not self.stop_event.is_set():
                try:
                    conn, peer = listener.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(
                    target=self._handle_connection,
                    args=(conn, peer),
                    daemon=True,
                )
                thread.start()
        finally:
            listener.close()
            if self._active_run_coordinator is not None:
                self._active_run_coordinator.stop()
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
            if (
                self.config.get("execution", {}).get("deployment") == "federated"
                and not self._experiment_terminal
            ):
                try:
                    current = self._federated.current_round if self._federated is not None else 0
                    self._checkpoint_manager.fail(
                        "Server stopped before validated experiment completion",
                        last_completed_round=int(current),
                    )
                except Exception:
                    pass
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

                elif op == "fl_policy_get":
                    self._handle_fl_policy_get(
                        conn,
                        request_id,
                        header,
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

    def _server_model_contract(self) -> Dict[str, Any]:
        if self._model_contract is None:
            workload = self._ensure_workload()
            self._model_contract = build_model_contract(
                self.config,
                workload.get_parameters(),
            )
            write_model_contract(self.config, self._model_contract)
        return self._model_contract

    def _validate_federated_model_contract(
        self,
        header: Dict[str, Any],
    ) -> None:
        received = header.get("model_contract")
        if not isinstance(received, dict):
            raise RuntimeError(
                "Federated request is missing model_contract. "
                "Use the same corrected code version on server and clients."
            )
        expected = compact_contract(self._server_model_contract())
        if str(received.get("contract_id", "")) != str(expected["contract_id"]):
            local = self._server_model_contract()
            received_id = str(received.get("contract_id", ""))
            raise RuntimeError(
                "FEDERATED MODEL CONTRACT MISMATCH: "
                f"server expects {local.get('family')}/"
                f"{local.get('architecture')}/{local.get('variant')} "
                f"({local.get('tensor_count')} tensors), while the client "
                f"reported {received.get('tensor_count')} tensors with "
                f"contract {received_id[:12]}.... "
                "Restart all FL participants with the same family, "
                "architecture, variant, framework, precision, input shape, "
                "and class count."
            )

    def _validate_federated_experiment_id(
        self,
        header: Dict[str, Any],
    ) -> None:
        expected = str(
            self.config["experiment"]["experiment_id"]
        )
        received = str(
            header.get("experiment_id", "")
        ).strip()
        if not received:
            raise RuntimeError(
                "Federated request is missing experiment_id"
            )
        if received != expected:
            raise RuntimeError(
                "Federated experiment ID mismatch: "
                f"server={expected!r}, client={received!r}. "
                "Restart the client with the same coordinated run ID."
            )

    def _validate_federated_training_policy(
        self,
        header: Dict[str, Any],
    ) -> None:
        expected = build_training_policy(self.config)
        received = str(header.get("training_policy_id", "")).strip()
        if not received:
            raise RuntimeError(
                "Federated request is missing training_policy_id. "
                "Clients must obtain and apply the server-authoritative "
                "training policy before requesting global weights."
            )
        if received != str(expected.get("policy_id")):
            raise RuntimeError(
                "FEDERATED TRAINING POLICY MISMATCH: client policy does not "
                "match the server-authoritative input size, batch size, "
                "learning rate, global rounds, local epochs, or local steps. "
                "Restart the client and accept the current server policy."
            )

    def _partition_index_for_client(self, client_id: str) -> int:
        expected = int(self.config.get("federated", {}).get("expected_clients", 1))
        value = str(client_id or "").strip()
        if not value:
            raise RuntimeError("Federated client_id is required")
        import re
        match = re.fullmatch(r"client_(\d+)", value, flags=re.IGNORECASE)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < expected:
                with self._ready_condition:
                    self._partition_slots[value] = index
                return index
        with self._ready_condition:
            existing = self._partition_slots.get(value)
            if existing is not None:
                return existing
            used = set(self._partition_slots.values())
            for index in range(expected):
                if index not in used:
                    self._partition_slots[value] = index
                    return index
        raise RuntimeError(f"No federated partition slot available for {value!r}")

    def _wait_for_all_clients_ready(self, client_id: str) -> tuple[int, int]:
        """Round-0 barrier: do not release the first model until all clients are ready."""
        expected = int(self.config.get("federated", {}).get("expected_clients", 1))
        value = str(client_id or "").strip()
        if not value:
            raise RuntimeError("Federated client_id is required for readiness")
        with self._ready_condition:
            before = len(self._ready_clients)
            self._ready_clients.add(value)
            ready = len(self._ready_clients)
            if ready != before:
                self.logger.write(
                    "federated_client_ready",
                    client_id=value,
                    ready_clients=ready,
                    expected_clients=expected,
                )
            if ready >= expected:
                self.logger.write(
                    "federated_all_clients_ready",
                    ready_clients=sorted(self._ready_clients),
                    expected_clients=expected,
                )
                self._ready_condition.notify_all()
            while len(self._ready_clients) < expected:
                self._ready_condition.wait()
            return len(self._ready_clients), expected

    def _handle_fl_policy_get(
        self,
        conn: socket.socket,
        request_id: str,
        header: Dict[str, Any],
    ) -> None:
        self._validate_federated_experiment_id(header)
        client_id = str(header.get("client_id", "")).strip()
        partition_index = self._partition_index_for_client(client_id)
        policy = build_training_policy(self.config)
        send_frame(
            conn,
            {
                "status": "ok",
                "request_id": request_id,
                "training_policy": policy,
                "run_id": self.config.get("experiment", {}).get("run_id"),
                "partition_index": partition_index,
                "ready_clients": len(self._ready_clients),
            },
        )
        self.logger.write(
            "federated_training_policy_sent",
            client_id=client_id,
            run_id=self.config.get("experiment", {}).get("run_id"),
            partition_index=partition_index,
            partition_type=policy.get("partition_type"),
            partition_alpha=policy.get("partition_alpha"),
            partition_seed=policy.get("partition_seed"),
            policy_id=policy.get("policy_id"),
            input_size=policy.get("input_size"),
            batch_size=policy.get("batch_size"),
            learning_rate=policy.get("learning_rate"),
            total_rounds=policy.get("rounds"),
            local_epochs=policy.get("local_epochs"),
            steps_per_epoch=policy.get("steps_per_epoch"),
        )

    def _handle_fl_get(
        self,
        conn: socket.socket,
        request_id: str,
        header: Dict[str, Any],
    ) -> None:
        self._validate_federated_experiment_id(header)
        self._validate_federated_training_policy(header)
        self._validate_federated_model_contract(header)
        # Block this request until every expected client has completed policy
        # synchronization and model construction. Operator startup delay is
        # therefore outside Round 0 and cannot inflate architecture timing.
        self._wait_for_all_clients_ready(str(header.get("client_id", "")))
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
        if not done:
            self._record_round_download(round_index, len(payload))

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
        self._validate_federated_experiment_id(header)
        self._validate_federated_training_policy(header)
        self._validate_federated_model_contract(header)
        coordinator = self._ensure_federated_coordinator()
        round_index = int(header["round"])
        client_id = str(header["client_id"])
        receive_timestamp = utc_now_iso()

        # recv_frame() has already received the full payload before this
        # handler is entered. Log this boundary before numpy deserialization
        # so the server timestamp is a clean network receive-completion mark.
        self.logger.write(
            "federated_phase",
            round=round_index,
            client_id=client_id,
            phase="Upload",
            boundary="receive_complete",
            bytes_received=len(payload),
            num_examples=int(
                header.get("num_examples", 1)
            ),
            client_loss=header.get("loss"),
            client_accuracy=header.get("accuracy"),
            client_precision=header.get("precision"),
            client_recall=header.get("recall"),
            client_f1=header.get("f1"),
            client_reconstruction_loss=header.get("reconstruction_loss"),
            client_mse=header.get("mse"),
            client_mae=header.get("mae"),
            client_kl_loss=header.get("kl_loss"),
        )

        parameters = bytes_to_arrays(payload)
        current_round, global_parameters, global_done = coordinator.get_global()
        if global_done or current_round != round_index:
            raise RuntimeError(
                f"Cannot evaluate client update for round {round_index}; "
                f"server current round is {current_round}"
            )
        client_model_norm = parameter_l2_norm(parameters)
        client_update_norm = parameter_delta_l2_norm(
            parameters, global_parameters
        )

        client_metrics = {
            "loss": header.get("loss"),
            "accuracy": header.get("accuracy"),
            "precision": header.get("precision"),
            "recall": header.get("recall"),
            "f1": header.get("f1"),
            "reconstruction_loss": header.get("reconstruction_loss"),
            "mse": header.get("mse"),
            "mae": header.get("mae"),
            "kl_loss": header.get("kl_loss"),
            "vae_beta": header.get("vae_beta"),
            "anomaly_accuracy": header.get("anomaly_accuracy"),
            "anomaly_precision": header.get("anomaly_precision"),
            "anomaly_recall": header.get("anomaly_recall"),
            "anomaly_f1": header.get("anomaly_f1"),
            "anomaly_auroc": header.get("anomaly_auroc"),
            "anomaly_auprc": header.get("anomaly_auprc"),
            "anomaly_threshold": header.get("anomaly_threshold"),
            "anomaly_tp": header.get("anomaly_tp"),
            "anomaly_fp": header.get("anomaly_fp"),
            "anomaly_tn": header.get("anomaly_tn"),
            "anomaly_fn": header.get("anomaly_fn"),
            "client_model_norm_l2": client_model_norm,
            "update_norm_l2": client_update_norm,
            "payload_bytes": len(payload),
            "partition_type": header.get("partition_type"),
            "partition_alpha": header.get("partition_alpha"),
            "partition_seed": header.get("partition_seed"),
            "partition_index": header.get("partition_index"),
            "partition_sample_count": header.get("partition_sample_count"),
            "partition_assignment_id": header.get("partition_assignment_id"),
        }

        if self._performance_writer is not None:
            self._performance_writer.write_server_client_update(
                {
                    "experiment_id": self.config["experiment"]["experiment_id"],
                    "round": round_index,
                    "client_id": client_id,
                    "num_examples": int(header.get("num_examples", 1)),
                    "payload_bytes": len(payload),
                    "partition_type": header.get("partition_type"),
                    "partition_alpha": header.get("partition_alpha"),
                    "partition_seed": header.get("partition_seed"),
                    "partition_index": header.get("partition_index"),
                    "partition_disjoint": True,
                    "partition_sample_count": header.get("partition_sample_count"),
                    "partition_assignment_id": header.get("partition_assignment_id"),
                    "client_loss": header.get("loss"),
                    "client_accuracy": header.get("accuracy"),
                    "client_precision": header.get("precision"),
                    "client_recall": header.get("recall"),
                    "client_f1": header.get("f1"),
                    "client_macro_precision": header.get("macro_precision"),
                    "client_macro_recall": header.get("macro_recall"),
                    "client_macro_f1": header.get("macro_f1"),
                    "client_reconstruction_loss": header.get("reconstruction_loss"),
                    "client_mse": header.get("mse"),
                    "client_mae": header.get("mae"),
                    "client_kl_loss": header.get("kl_loss"),
                    "vae_beta": header.get("vae_beta"),
                    "client_anomaly_accuracy": header.get("anomaly_accuracy"),
                    "client_anomaly_precision": header.get("anomaly_precision"),
                    "client_anomaly_recall": header.get("anomaly_recall"),
                    "client_anomaly_f1": header.get("anomaly_f1"),
                    "client_anomaly_auroc": header.get("anomaly_auroc"),
                    "client_anomaly_auprc": header.get("anomaly_auprc"),
                    "client_anomaly_threshold": header.get("anomaly_threshold"),
                    "client_anomaly_tp": header.get("anomaly_tp"),
                    "client_anomaly_fp": header.get("anomaly_fp"),
                    "client_anomaly_tn": header.get("anomaly_tn"),
                    "client_anomaly_fn": header.get("anomaly_fn"),
                    "client_model_norm_l2": client_model_norm,
                    "client_update_norm_l2": client_update_norm,
                    "receive_timestamp_utc": receive_timestamp,
                }
            )

        sync_wait_start = time.perf_counter_ns()

        next_round, done = coordinator.submit_update(
            round_index=round_index,
            client_id=client_id,
            parameters=parameters,
            num_examples=int(
                header.get("num_examples", 1)
            ),
            metrics=client_metrics,
        )

        sync_wait_end = time.perf_counter_ns()
        self.logger.write(
            "federated_sync_wait",
            round=round_index,
            client_id=client_id,
            wait_time_ms=(
                sync_wait_end - sync_wait_start
            ) / 1_000_000.0,
            next_round=next_round,
            done=done,
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
        if done:
            with self._ready_condition:
                self._final_ack_clients.add(client_id)
                expected = int(self.config.get("federated", {}).get("expected_clients", 1))
                if len(self._final_ack_clients) >= expected:
                    self.logger.write(
                        "federated_training_complete",
                        completed_rounds=int(self.config["federated"]["rounds"]),
                        clients=sorted(self._final_ack_clients),
                    )
                    metrics_complete = True
                    if self._performance_enabled:
                        output_dir = Path(self.config["experiment"]["output_dir"])
                        metrics_path = output_dir / "round_metrics.csv"
                        updates_path = output_dir / "client_update_metrics.csv"
                        try:
                            with metrics_path.open("r", encoding="utf-8") as handle:
                                metrics_rows = max(sum(1 for _ in handle) - 1, 0)
                            with updates_path.open("r", encoding="utf-8") as handle:
                                update_rows = max(sum(1 for _ in handle) - 1, 0)
                            target_rounds = int(self.config["federated"]["rounds"])
                            expected_clients = int(self.config["federated"]["expected_clients"])
                            metrics_complete = (
                                metrics_rows >= target_rounds
                                and update_rows >= target_rounds * expected_clients
                            )
                        except OSError:
                            metrics_complete = False
                    self._checkpoint_manager.complete(metrics_complete=metrics_complete)
                    self._experiment_completed = bool(metrics_complete)
                    self._experiment_terminal = True
                    self.stop_event.set()

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
