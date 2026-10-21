#!/usr/bin/env python3
"""
Run a planned campaign unattended from the DGX, one schedule row at a time.

For each row that has not passed yet:
  1. start the server here, then the proxy (--wait-for-run) and the three clients over SSH
  2. wait for the server to finish (fail on a stall or timeout)
  3. stop the proxy with SIGINT (Ctrl+C) so it finishes the capture and uploads its results
  4. pull the pcap chunks from the proxy, extract them in parallel and run check_run.py
  5. PASS: delete the chunks on the proxy laptop; FAIL: delete the attempt's outputs on every
     machine and retry (--attempts), then move on and leave the row marked FAILED

Progress is kept in <campaign>/runner_state.json, so the runner can be stopped and restarted;
touch <campaign>/STOP to stop after the current run. Logs go to --log-dir.

Usage
  nohup python3 campaign_runner.py --campaign campaigns/custom_main > runner.log 2>&1 &
"""
import argparse, csv, json, os, re, shlex, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
SERVER_PY = "/home/dgx-1/anaconda3/envs/fingerprinting/bin/python"
TOOLS_PY = sys.executable
PCAP_ROOT = Path("/home/dgx-1/pcaps")
OFFLINE = "HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
       "-o", "ServerAliveCountMax=6"]
PROXY = {"ssh": "lawrence@10.42.0.1", "repo": "~/Lawrence/fingerprinting", "py": "python3"}
LINUX_CLIENTS = {
    "client_1": {"ssh": "anik@10.42.0.210", "repo": "~/Lawrence/Fingerprinting", "py": "~/aifp/bin/python3"},
    "client_2": {"ssh": "tiger@10.42.0.47", "repo": "~/Lawrence/Fingerprinting",
                 "py": "/home/tiger/anaconda3/envs/aifp/bin/python"},
}
WIN_CLIENT = {"id": "client_3", "ssh": "oyala@10.42.0.145", "repo": r"C:\Users\oyala\Desktop\openAIFinger",
              "py": r"C:\Users\oyala\aifp\Scripts\python.exe"}
STALL_SEC = 45 * 60
# Windows processes running main.py (not every python.exe, e.g. editors' language servers)
WIN_FIND_MAIN = ("powershell -NoProfile -Command \"Get-CimInstance Win32_Process | Where-Object "
                 "{ $_.Name -eq 'python.exe' -and $_.CommandLine -like '*main.py*' } | "
                 "Select-Object -ExpandProperty ProcessId\"")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def ssh(target, cmd, timeout=120, check=False):
    r = subprocess.run(SSH + [target, cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{target}: {cmd!r} failed: {r.stderr.strip()[-300:]}")
    return r


def detached(target, repo, py, args, logfile):
    inner = f"export {OFFLINE}; exec {py} -u main.py {args} > {logfile} 2>&1"
    ssh(target, f"cd {repo} && mkdir -p ~/runlogs && setsid -f sh -c {shlex.quote(inner)} < /dev/null", check=True)


class Runner:
    def __init__(self, a):
        self.a = a
        self.camp = (REPO / a.campaign).resolve()
        self.state_path = self.camp / "runner_state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.logs = Path(a.log_dir)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.rows = list(csv.DictReader(open(self.camp / "schedule.csv")))

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def entry(self, order):
        return self.state.setdefault(str(order), {"status": "planned", "attempts": []})

    # ---- one attempt ------------------------------------------------------------------
    def run_dir_of(self, row):
        cfg = yaml.safe_load(open(REPO / row["config_dir"] / "server.yaml"))
        return cfg["experiment"]["run_dir"]

    def busy(self):
        out = []
        for name, t in [("proxy", PROXY["ssh"])] + [(c, v["ssh"]) for c, v in LINUX_CLIENTS.items()]:
            if ssh(t, "pgrep -f 'main.py (run|proxy)'").stdout.strip():
                out.append(name)
        if ssh(WIN_CLIENT["ssh"], WIN_FIND_MAIN).stdout.strip():
            out.append(WIN_CLIENT["id"])
        if subprocess.run(["pgrep", "-f", "main.py run --config"], capture_output=True).stdout.strip():
            out.append("server")
        return out

    def cleanup(self, row, run_id=None):
        """Delete one attempt's outputs on every machine (the user prefers deleting failed attempts)."""
        rd = self.run_dir_of(row)
        subprocess.run(["rm", "-rf", str(REPO / rd)])
        for v in LINUX_CLIENTS.values():
            ssh(v["ssh"], f"cd {v['repo']} && rm -rf {shlex.quote(rd)}")
        win = rd.replace("/", "\\")
        ssh(WIN_CLIENT["ssh"], f'cd /d {WIN_CLIENT["repo"]} && if exist {win} rmdir /s /q {win}')
        if run_id:
            subprocess.run(["rm", "-rf", str(REPO / "collected_experiments" / run_id), str(PCAP_ROOT / run_id)])
            ssh(PROXY["ssh"], f"cd {PROXY['repo']} && rm -rf experiments/staging/{run_id}")

    def attempt(self, row):
        order, cfgdir = int(row["order"]), row["config_dir"]
        tag = f"{order:03d}"
        rd = REPO / self.run_dir_of(row)
        progress = rd / "server" / "round_progress.json"
        status_json = rd / "server" / "experiment_status.json"
        est = float(row["est_hours"] or 3.0)
        limit = (est * 3 + 1.5) * 3600
        res = {"started": now()}

        env = dict(os.environ, HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        server = subprocess.Popen([SERVER_PY, "-u", "main.py", "run", "--config", f"{cfgdir}/server.yaml"],
                                  cwd=REPO, env=env, stdout=open(self.logs / f"{tag}_server.log", "w"),
                                  stderr=subprocess.STDOUT, start_new_session=True)
        win = None
        self.current_run_id = None
        try:
            t0 = time.time()
            while not (status_json.exists() and "run_id" in status_json.read_text()):
                if server.poll() is not None or time.time() - t0 > 300:
                    raise RuntimeError("server did not start")
                time.sleep(2)
            time.sleep(3)
            run_id = json.loads(status_json.read_text())["run_id"]
            res["run_id"] = self.current_run_id = run_id
            log(f"{tag} server up, {run_id}")

            detached(PROXY["ssh"], PROXY["repo"], PROXY["py"], "proxy --config proxy.yaml --wait-for-run",
                     f"~/runlogs/{tag}_proxy.log")
            t0 = time.time()
            while run_id not in ssh(PROXY["ssh"], "pgrep -a dumpcap").stdout:
                if time.time() - t0 > 180:
                    raise RuntimeError("proxy did not start capturing for this run")
                time.sleep(3)
            log(f"{tag} proxy capturing")

            for cid, v in LINUX_CLIENTS.items():
                detached(v["ssh"], v["repo"], v["py"], f"run --config {cfgdir}/{cid}.yaml", f"~/runlogs/{tag}_{cid}.log")
            wcfg = cfgdir.replace("/", "\\") + "\\client_3.yaml"
            wcmd = (f'cd /d {WIN_CLIENT["repo"]} && set "HF_DATASETS_OFFLINE=1" && set "HF_HUB_OFFLINE=1" && '
                    f'set "TRANSFORMERS_OFFLINE=1" && {WIN_CLIENT["py"]} -u main.py run --config {wcfg}')
            win = subprocess.Popen(SSH + [WIN_CLIENT["ssh"], wcmd], stdout=open(self.logs / f"{tag}_client_3.log", "w"),
                                   stderr=subprocess.STDOUT, start_new_session=True)
            log(f"{tag} clients started")

            t0 = last_change = time.time()
            last = None
            while server.poll() is None:
                time.sleep(20)
                cur = progress.read_text() if progress.exists() else None
                done = json.loads(cur)["last_completed_round"] if cur else 0
                if done != last:
                    last, last_change = done, time.time()
                if time.time() - last_change > STALL_SEC:
                    raise RuntimeError(f"no round completed for {STALL_SEC // 60} min (at round {done})")
                if time.time() - t0 > limit:
                    raise RuntimeError(f"run exceeded {limit / 3600:.1f} h")
                if win.poll() not in (None, 0):
                    raise RuntimeError(f"Windows client exited with {win.returncode}")
            final = json.loads(progress.read_text()) if progress.exists() else {}
            res["server_rc"] = server.returncode
            res["rounds"] = final.get("last_completed_round")
            log(f"{tag} server finished rc={server.returncode} rounds={res['rounds']}")

            t0 = time.time()
            while time.time() - t0 < 900:
                left = [c for c, v in LINUX_CLIENTS.items()
                        if ssh(v["ssh"], f"pgrep -f {shlex.quote(cfgdir + '/' + c)}").stdout.strip()]
                if win.poll() is None:
                    left.append(WIN_CLIENT["id"])
                if not left:
                    break
                time.sleep(10)
            else:
                raise RuntimeError(f"clients still running 15 min after the server: {left}")
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(120)
                except subprocess.TimeoutExpired:
                    server.kill()
            for cid, v in LINUX_CLIENTS.items():
                ssh(v["ssh"], f"pkill -INT -f {shlex.quote(cfgdir + '/' + cid)}")
            if win is not None and win.poll() is None:
                for pid in ssh(WIN_CLIENT["ssh"], WIN_FIND_MAIN).stdout.split():
                    ssh(WIN_CLIENT["ssh"], f"taskkill /F /T /PID {pid}")
                win.kill()
            self.stop_proxy()

        coll = REPO / "collected_experiments" / run_id
        t0 = time.time()
        while not all((coll / r / "experiment_status.json").exists()
                      for r in ("server", "client_1", "client_2", "client_3", "proxy")):
            if time.time() - t0 > 600:
                raise RuntimeError("not every role reached the collector")
            time.sleep(10)

        dest = PCAP_ROOT / run_id
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--partial", "-e", " ".join(SSH),
                        f"{PROXY['ssh']}:{PROXY['repo'].replace('~/', '')}/experiments/staging/{run_id}/proxy/pcap/",
                        f"{dest}/"], check=True)
        man = json.loads((coll / "proxy" / "capture_chunks_manifest.json").read_text())
        bad = [c["path"] for c in man["chunks"]
               if not (dest / Path(c["path"]).name).exists() or (dest / Path(c["path"]).name).stat().st_size != c["size_bytes"]]
        if bad:
            raise RuntimeError(f"pcap copy incomplete: {bad}")
        res["capture_gb"] = round(man["total_capture_bytes"] / 1e9, 2)
        log(f"{tag} copied {res['capture_gb']} GB, extracting")

        with open(self.logs / f"{tag}_check.log", "w") as fh:
            subprocess.run([TOOLS_PY, "extract_pcaps_parallel.py", str(coll), "--pcap-dir", str(dest)],
                           cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)
            rc = subprocess.run([TOOLS_PY, "check_run.py", str(coll), "--plan", str(self.camp / "schedule.csv")],
                                cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
        res["check"] = "PASS" if rc == 0 else "FAIL"
        if rc == 0:
            ssh(PROXY["ssh"], f"cd {PROXY['repo']} && rm -rf experiments/staging/{run_id}/proxy/pcap")
        return res

    def stop_proxy(self):
        ssh(PROXY["ssh"], "pkill -INT -f 'main.py proxy'")
        t0 = time.time()
        while ssh(PROXY["ssh"], "pgrep -f 'main.py proxy'").stdout.strip():
            if time.time() - t0 > 900:
                ssh(PROXY["ssh"], "pkill -TERM -f 'main.py proxy'")
                break
            time.sleep(5)

    # ---- campaign loop ----------------------------------------------------------------
    def main(self):
        log(f"campaign {self.camp.name}: {len(self.rows)} rows, state {self.state_path.name}")
        for row in self.rows:
            order = int(row["order"])
            e = self.entry(order)
            if e["status"] in ("PASS", "FAILED", "MANUAL"):
                continue
            if (self.camp / "STOP").exists():
                log("STOP file found; stopping before the next run")
                return
            busy = self.busy()
            if busy:
                log(f"machines busy before {order:03d}: {busy}; stopping so nothing is mixed")
                return
            while e["status"] not in ("PASS", "FAILED"):
                n = len(e["attempts"]) + 1
                log(f"{order:03d} {row['variant']} attempt {n}")
                self.cleanup(row)
                try:
                    res = self.attempt(row)
                except Exception as exc:
                    res = {"error": f"{type(exc).__name__}: {exc}", "run_id": getattr(self, "current_run_id", None)}
                    log(f"{order:03d} attempt {n} error: {res['error']}")
                res["ended"] = now()
                e["attempts"].append(res)
                if res.get("check") == "PASS":
                    e.update(status="PASS", run_id=res["run_id"])
                elif n >= self.a.attempts:
                    e["status"] = "FAILED"
                    self.cleanup(row, res.get("run_id"))
                else:
                    self.cleanup(row, res.get("run_id"))
                self.save()
                log(f"{order:03d} -> {e['status']}")
        log("campaign complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign", default="campaigns/custom_main")
    ap.add_argument("--attempts", type=int, default=2, help="attempts per row before it is marked FAILED")
    ap.add_argument("--log-dir", default="/home/dgx-1/runlogs")
    Runner(ap.parse_args()).main()
