"""
NVIDIA FLARE backend: the same models, data partitions, FedAvg, monitoring and output layout as
the custom backend, with FLARE's gRPC transport (plaintext, TLS or mutual TLS) on the wire.

Selected with `federated.fl_framework: nvidia_flare`.

FLARE needs a provisioned startup kit per site, made once per testbed:

    python3 main.py flare-provision --out flare_kits --server-host 10.42.0.195 \\
        --proxy-host 10.42.0.1 --proxy-port 8080 --clients client_1 client_2 client_3 --security mtls

Clients are provisioned to connect to the proxy (connect_to = proxy host and port), so their
traffic passes the observation point; the server listens on the proxy's upstream port, and the
admin console connects to the server directly. Copy flare_kits/<project>/prod_00/<site> to each
machine and set `federated.flare.kits_dir` to the prod_00 folder in every role's config.

Server role: starts the FLARE server from its kit and the run coordinator, waits for all
expected clients, submits a FedAvg job whose clients run ai_fingerprint/flare_client_job.py,
waits for it to finish, shuts the system down, and copies FLARE's logs to the output folder.
Client role: starts the FLARE client from its kit and waits until the server shuts it down.

Requires the optional package: pip install "nvflare==2.6.3"
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .experiment_coordination import ActiveRunCoordinator
from .experiment_layout import materialize_role_metadata
from .metadata import EventLogger
from .resource_monitor import ResourceMonitor

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_NAME = "flare-server"
ADMIN_NAME = "admin@aifp.lab"


def _require_flare():
    try:
        import nvflare  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('federated.fl_framework=nvidia_flare requires: pip install "nvflare==2.6.3"') from exc


# ---------------------------------------------------------------- provisioning
def project_yaml(name: str, server_host: str, fl_port: int, admin_port: int, proxy_host: str,
                 proxy_port: int, clients: List[str], security: str) -> Dict[str, Any]:
    server = {"name": SERVER_NAME, "type": "server", "org": "lab", "fed_learn_port": fl_port,
              "admin_port": admin_port, "default_host": server_host, "host_names": [server_host, "localhost"],
              "connection_security": security}
    parts = [server]
    for c in clients:
        parts.append({"name": c, "type": "client", "org": "lab",
                      "connect_to": {"host": proxy_host, "port": proxy_port}})
    parts.append({"name": ADMIN_NAME, "type": "admin", "org": "lab", "role": "project_admin"})
    return {
        "api_version": 3, "name": name, "description": "AI fingerprinting FL testbed",
        "participants": parts,
        "builders": [
            {"path": "nvflare.lighter.impl.workspace.WorkspaceBuilder"},
            {"path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
             "args": {"config_folder": "config",
                      "overseer_agent": {"path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent",
                                         "overseer_exists": False,
                                         "args": {"sp_end_point": f"{SERVER_NAME}:{fl_port}:{admin_port}"}}}},
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
        ],
    }


def provision(out: str, name: str, server_host: str, fl_port: int, admin_port: int, proxy_host: str,
              proxy_port: int, clients: List[str], security: str) -> Path:
    _require_flare()
    out_dir = Path(out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    proj = out_dir / "project.yml"
    proj.write_text(yaml.safe_dump(project_yaml(name, server_host, fl_port, admin_port, proxy_host,
                                                proxy_port, clients, security), sort_keys=False))
    nvflare_cli = Path(sys.executable).with_name("nvflare")
    cmd = [str(nvflare_cli) if nvflare_cli.exists() else "nvflare", "provision", "-p", str(proj), "-w", str(out_dir)]
    subprocess.run(cmd, check=True)
    prods = sorted((out_dir / name).glob("prod_*"))
    if not prods:
        raise RuntimeError(f"provisioning produced no prod_* folder under {out_dir / name}")
    return prods[-1]


# ---------------------------------------------------------------- shared helpers
def _kits_dir(config: Dict[str, Any]) -> Path:
    kits = ((config.get("federated", {}) or {}).get("flare", {}) or {}).get("kits_dir")
    if not kits:
        raise RuntimeError("federated.flare.kits_dir must point to the provisioned prod_* folder")
    path = Path(kits).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"federated.flare.kits_dir {path} does not exist")
    return path


def _effective_config(config: Dict[str, Any]) -> Path:
    """Role config with an absolute output folder, read by the job processes FLARE spawns."""
    cfg = dict(config)
    cfg["experiment"] = dict(config["experiment"])
    out = Path(config["experiment"]["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg["experiment"]["output_dir"] = str(out)
    path = out / "flare_role_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _env(config_path: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["AIFP_FLARE_CONFIG"] = str(config_path)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _site_command(kit: Path, module: str, spec: str, extra: str) -> List[str]:
    return [sys.executable, "-u", "-m", module, "-m", str(kit), "-s", spec, "--set", "secure_train=true",
            *extra.split(), "org=lab", "config_folder=config"]


def _copy_logs(kit: Path, out: Path, tag: str) -> None:
    for p in kit.rglob("log.txt"):
        rel = p.relative_to(kit)
        dest = out / "flare_logs" / tag / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)


# ---------------------------------------------------------------- roles
class FlareServer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = EventLogger(config)

    def serve_forever(self) -> None:
        _require_flare()
        from nvflare.fuel.flare_api.flare_api import new_secure_session
        from nvflare.job_config.api import FedJob
        from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

        from .flare_components import LoggingFedAvg, WorkloadPersistor

        cfg, fed = self.config, self.config["federated"]
        kits = _kits_dir(cfg)
        kit, admin_kit = kits / SERVER_NAME, kits / ADMIN_NAME
        n, rounds = int(fed["expected_clients"]), int(fed["rounds"])
        coordinator = None
        if bool((cfg.get("coordination", {}) or {}).get("enabled", True)):
            coordinator = ActiveRunCoordinator(cfg)
            coordinator.start()
            print(f"[server] experiment coordinator: {coordinator.host}:{coordinator.port} "
                  f"run_id={coordinator.run_id}")
        materialize_role_metadata(cfg)
        self.logger.refresh_base(cfg)
        role_cfg = _effective_config(cfg)
        out = Path(role_cfg).parent
        monitor = ResourceMonitor(cfg)
        self.logger.write("experiment_start", fl_framework="nvidia_flare", rounds=rounds, expected_clients=n)
        monitor.start()
        proc = subprocess.Popen(_site_command(kit, "nvflare.private.fed.app.server.server_train",
                                              "fed_server.json", ""),
                                env=_env(role_cfg), stdout=open(out / "flare_server_stdout.log", "w"),
                                stderr=subprocess.STDOUT)
        session = None
        try:
            session = self._connect(new_secure_session, admin_kit, proc)
            print(f"[server] FLARE server up; waiting for {n} clients")
            deadline = time.time() + float(fed.get("flare", {}).get("client_wait_sec", 1800))
            while len(session.get_connected_client_list()) < n:
                if time.time() > deadline or proc.poll() is not None:
                    raise RuntimeError("clients did not connect to the FLARE server in time")
                time.sleep(5)
            job = FedJob(name=f"aifp_{cfg['experiment']['experiment_id']}", min_clients=n)
            job.to_server(WorkloadPersistor(), id="persistor")
            job.to_server(LoggingFedAvg(num_clients=n, num_rounds=rounds, persistor_id="persistor"))
            job.to_clients(ScriptRunner(script=str(REPO_ROOT / "ai_fingerprint" / "flare_client_job.py"),
                                        framework=FrameworkType.NUMPY))
            with tempfile.TemporaryDirectory() as tmp:
                job.export_job(tmp)
                job_id = session.submit_job(str(Path(tmp) / job.name))
            print(f"[server] submitted FedAvg job {job_id}: {rounds} rounds, {n} clients")
            self.logger.write("flare_job_submitted", job_id=job_id, rounds=rounds, clients=n)
            status = session.monitor_job(job_id)
            self.logger.write("flare_job_finished", job_id=job_id, status=str(status))
            print(f"[server] job {job_id} finished: {status}")
        finally:
            if session is not None:
                # shutdown_system() needs an overseer, which this deployment does not run; shutting
                # down "all" stops every connected client and then the server.
                from nvflare.fuel.flare_api.api_spec import TargetType
                try:
                    session.shutdown(TargetType.ALL)
                except Exception as exc:
                    self.logger.write("flare_shutdown_error", error=f"{type(exc).__name__}: {exc}")
                session.close()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.terminate()
            monitor.stop()
            _copy_logs(kit, out, "server")
            self.logger.write("experiment_stop", fl_framework="nvidia_flare")
            if coordinator is not None:
                coordinator.stop()

    @staticmethod
    def _connect(new_secure_session, admin_kit: Path, proc) -> Any:
        last = None
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError("FLARE server exited during start-up; see flare_server_stdout.log")
            try:
                return new_secure_session(ADMIN_NAME, str(admin_kit), timeout=10)
            except Exception as exc:  # server not up yet
                last = exc
                time.sleep(5)
        raise RuntimeError(f"could not connect to the FLARE server: {last}")


class FlareClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = EventLogger(config)

    def run(self) -> None:
        _require_flare()
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
        kit = _kits_dir(cfg) / client_id
        if not kit.is_dir():
            raise RuntimeError(f"no FLARE startup kit for {client_id} in {kit.parent}")
        from .flower_backend import adopt_active_run_id
        adopt_active_run_id(cfg)
        materialize_role_metadata(cfg)
        self.logger.refresh_base(cfg)
        role_cfg = _effective_config(cfg)
        out = Path(role_cfg).parent
        from .flower_backend import _local_ip_towards
        host, port = str(cfg["node"]["host"]), int(cfg["node"]["port"])
        self.logger.write("network_registration", client_id=client_id,
                          local_ip=_local_ip_towards(host, port), local_port=None,
                          remote_host=host, remote_port=port, transport="flare_grpc",
                          fl_framework="nvidia_flare")
        monitor = ResourceMonitor(cfg)
        self.logger.write("experiment_start", fl_framework="nvidia_flare")
        monitor.start()
        print(f"[client] {client_id} starting FLARE client from {kit}")
        try:
            rc = subprocess.call(_site_command(kit, "nvflare.private.fed.app.client.client_train",
                                               "fed_client.json", f"uid={client_id}"),
                                 env=_env(role_cfg), stdout=open(out / "flare_client_stdout.log", "w"),
                                 stderr=subprocess.STDOUT)
        finally:
            monitor.stop()
            _copy_logs(kit, out, client_id)
            self.logger.write("experiment_stop", fl_framework="nvidia_flare")
        if rc not in (0, None):
            raise RuntimeError(f"FLARE client exited with code {rc}; see flare_client_stdout.log")
