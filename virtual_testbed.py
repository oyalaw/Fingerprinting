#!/usr/bin/env python3
"""
Virtual testbed: proxy, FL server and three clients as isolated network namespaces on one Linux
machine, running the real code (main.py run / main.py proxy) and a real packet capture.

No root or Docker is needed: the script re-runs itself inside an unprivileged user + network
namespace, where it may create virtual links and capture packets that never leave the machine.

Topology (addresses as on the physical testbed)

    client_1 10.42.0.210 ─┐
    client_2 10.42.0.47  ─┼─ br0 10.42.0.1 (proxy: relay + dumpcap) ── server 10.42.0.195
    client_3 10.42.0.145 ─┘

  * every role has its own namespace and virtual Ethernet link (MTU 1500, offloads off, so
    packets look like the hotspot capture), shaped with tc netem (rate, delay, jitter)
  * clients differ like devices do: CPU cores (taskset + thread limits) and link profile
  * the proxy captures client-facing traffic on br0; the result collector runs on the server
  * a run = server, then proxy (--wait-for-run), then clients; afterwards the proxy is stopped
    with SIGINT, the capture is extracted in parallel and check_run.py verifies the run

Limits: all roles share one kernel and clock, the "devices" are CPU-limited processes on the
same machine (no GPU, no Jetson/Windows behaviour), and the links are emulated, not Wi-Fi.

Usage
  python3 virtual_testbed.py --plan campaigns/vlab --out vlab_runs [--rounds 15] [--only resnet18]
"""
import argparse, csv, glob, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
NET = "10.42.0"
ROLES = {
    # role: (ip, cpu cores, rate, delay, jitter)  -- client profiles stand in for the devices
    "server":   (f"{NET}.195", "14-17", "1gbit",   "0.5ms", "0.1ms"),
    "client_1": (f"{NET}.210", "0-1",   "100mbit", "3ms",   "1ms"),    # Jetson-like: 2 cores, slow link
    "client_2": (f"{NET}.47",  "2-9",   "400mbit", "1ms",   "0.3ms"),  # desktop-like: 8 cores
    "client_3": (f"{NET}.145", "10-13", "200mbit", "2ms",   "0.6ms"),  # laptop-like: 4 cores
}
PROXY_IP, PROXY_CORES = f"{NET}.1", "18-19"


def sh(cmd, check=True, **kw):
    return subprocess.run(cmd, check=check, capture_output=True, text=True, **kw)


def ncores(spec):
    a, _, b = spec.partition("-")
    return int(b or a) - int(a) + 1


# ------------------------------------------------------------------ topology
class Lab:
    def __init__(self):
        self.ns = {}

    def up(self):
        sh(["ip", "link", "set", "lo", "up"])
        sh(["ip", "link", "add", "br0", "type", "bridge"])
        sh(["ip", "addr", "add", f"{PROXY_IP}/24", "dev", "br0"])
        sh(["ip", "link", "set", "br0", "up"])
        for role, (ip, _, rate, delay, jitter) in ROLES.items():
            p = subprocess.Popen(["unshare", "--net", "sleep", "infinity"])
            time.sleep(0.2)
            self.ns[role] = p
            host, peer = f"v_{role}"[:15], "eth0"
            sh(["ip", "link", "add", host, "type", "veth", "peer", "name", peer, "netns", str(p.pid)])
            sh(["ip", "link", "set", host, "master", "br0", "up"])
            self._offloads_off(host)
            self.run_in(role, ["ip", "link", "set", "lo", "up"])
            self.run_in(role, ["ip", "addr", "add", f"{ip}/24", "dev", peer])
            self.run_in(role, ["ip", "link", "set", peer, "up"])
            self.run_in(role, ["ethtool", "-K", peer, "tso", "off", "gso", "off", "gro", "off"], check=False)
            netem = ["netem", "rate", rate, "delay", delay, jitter]
            self.run_in(role, ["tc", "qdisc", "add", "dev", peer, "root", *netem])   # role -> network
            sh(["tc", "qdisc", "add", "dev", host, "root", *netem])                 # network -> role
        self._offloads_off("br0")

    @staticmethod
    def _offloads_off(dev):
        sh(["ethtool", "-K", dev, "tso", "off", "gso", "off", "gro", "off"], check=False)

    def prefix(self, role):
        return ["nsenter", f"--net=/proc/{self.ns[role].pid}/ns/net"]

    def run_in(self, role, cmd, check=True):
        return sh(self.prefix(role) + cmd, check=check)

    def spawn(self, role, cmd, log, cores):
        # The lab has no internet: Hugging Face datasets must use the local cache only (it otherwise
        # contacts huggingface.co even for cached data and stalls on retries).
        env = dict(os.environ, PYTHONUNBUFFERED="1", AIFP_SKIP_DEVICE_CHECK="1",
                   HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   AIFP_NODE_IDENTITY=str(REPO / ".vlab_no_identity.yaml"))
        n = str(ncores(cores))
        env.update(OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, TORCH_NUM_THREADS=n)
        pre = self.prefix(role) if role in self.ns else []
        return subprocess.Popen(pre + ["taskset", "-c", cores] + cmd, cwd=REPO, env=env,
                                stdout=open(log, "w"), stderr=subprocess.STDOUT)

    def down(self):
        for p in self.ns.values():
            p.kill()


# ------------------------------------------------------------------ configs
def role_config(src, role, rounds, out):
    cfg = yaml.safe_load(open(src))
    ip = ROLES[role][0]
    cfg["federated"].update({"mode": "smoke_test" if rounds < 50 else "full_scale", "rounds": rounds})
    cfg["coordination"].update({"enabled": True, "host": ROLES["server"][0], "port": 8081})
    cfg["result_collection"].update({"enabled": True, "collector_host": ROLES["server"][0], "collector_port": 8090})
    cfg["node"].update({"host": ip if role == "server" else PROXY_IP, "port": 8080})
    cfg.setdefault("resource_monitor", {})["network_interface"] = "eth0"
    exp = cfg["experiment"]
    for key in ("output_dir", "run_dir", "branch_dir"):
        if exp.get(key):
            exp[key] = str(out / "experiments" / Path(exp[key]).relative_to(exp["results_root"]))
    exp["results_root"] = str(out / "experiments")
    exp["run_id"] = None
    cfg.pop("campaign", None)
    return cfg


def proxy_config(template, out):
    cfg = yaml.safe_load(open(template))
    cfg["experiment"] = {"experiment_id": "auto", "results_root": str(out / "staging"),
                         "existing_output_policy": "error"}
    cfg["coordination"] = {"enabled": True, "server_host": ROLES["server"][0], "server_port": 8081}
    cfg["proxy"].update({"listen_host": PROXY_IP, "listen_port": 8080,
                         "upstream_host": ROLES["server"][0], "upstream_port": 8080})
    cap = cfg["capture"]
    cap.update({"enabled": True, "interface": "br0", "extract_after": False, "client_discovery_mode": "automatic",
                "client_ips": [], "client_aliases": {}, "minimum_free_disk_gb": 5.0})
    cap["offload_management"] = {"enabled": False, "required": False}
    cfg["architecture_inference"] = {"enabled": False, "realtime_enabled": False, "final_enabled": False,
                                     "realtime_required": False}
    cfg["result_collection"].update({"enabled": True, "collector_host": ROLES["server"][0], "collector_port": 8090})
    return cfg


def wait_for(path_or_fn, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (path_or_fn() if callable(path_or_fn) else Path(path_or_fn).exists()):
            return True
        time.sleep(1)
    return False


# ------------------------------------------------------------------ one run
def run_one(lab, run_dir, out, rounds, py, proxy_tpl, timeout):
    name = run_dir.name
    cdir, logs = out / "configs" / name, out / "logs" / name
    cdir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    paths = {}
    for role in ROLES:
        paths[role] = cdir / f"{role}.yaml"
        yaml.safe_dump(role_config(run_dir / f"{role}.yaml", role, rounds, out), open(paths[role], "w"), sort_keys=False)
    pcfg = cdir / "proxy.yaml"
    yaml.safe_dump(proxy_config(proxy_tpl, out), open(pcfg, "w"), sort_keys=False)
    t0 = time.time()
    procs = {"server": lab.spawn("server", [py, "main.py", "run", "--config", str(paths["server"])],
                                 logs / "server.log", ROLES["server"][1])}
    wait_for(lambda: "experiment coordinator" in (logs / "server.log").read_text(errors="ignore"), 120)
    procs["proxy"] = lab.spawn("proxy", [py, "main.py", "proxy", "--config", str(pcfg), "--wait-for-run"],
                               logs / "proxy.log", PROXY_CORES)
    wait_for(lambda: "press Ctrl+C" in (logs / "proxy.log").read_text(errors="ignore"), 120)
    time.sleep(2)
    for c in ("client_1", "client_2", "client_3"):
        procs[c] = lab.spawn(c, [py, "main.py", "run", "--config", str(paths[c])], logs / f"{c}.log", ROLES[c][1])
    codes = {}
    for c in ("client_1", "client_2", "client_3"):
        try:
            codes[c] = procs[c].wait(timeout=max(5, timeout - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            procs[c].kill()
            codes[c] = "timeout"
    try:
        codes["server"] = procs["server"].wait(timeout=180)
    except subprocess.TimeoutExpired:
        procs["server"].kill()
        codes["server"] = "server_stuck"
    procs["proxy"].send_signal(signal.SIGINT)
    try:
        codes["proxy"] = procs["proxy"].wait(timeout=600)
    except subprocess.TimeoutExpired:
        procs["proxy"].kill()
        codes["proxy"] = "proxy_stuck"
    run_id = None
    for line in (logs / "proxy.log").read_text(errors="ignore").splitlines():
        if "[proxy] active run " in line:
            run_id = line.split("[proxy] active run ")[1].split(";")[0].strip()
    row = {"run": name, "run_id": run_id, "seconds_fl": round(time.time() - t0, 1),
           **{f"rc_{k}": v for k, v in codes.items()}}
    cfg = yaml.safe_load(open(paths["server"]))
    row.update(variant=cfg["ai"]["variant"], family=cfg["ai"]["family"], dataset=cfg["ai"]["dataset"])
    collected = out / "collected" / str(run_id)
    pcaps = out / "staging" / str(run_id) / "proxy" / "pcap"
    if run_id and wait_for(lambda: (collected / "proxy").exists() and (collected / "server").exists(), 120) \
            and pcaps.exists():
        ex = sh([py, "extract_pcaps_parallel.py", str(collected), "--pcap-dir", str(pcaps), "--workers", "8"],
                check=False, cwd=REPO)
        (logs / "extract.log").write_text(ex.stdout + ex.stderr)
        ck = sh([py, "check_run.py", str(collected)], check=False, cwd=REPO)
        (logs / "check.log").write_text(ck.stdout + ck.stderr)
        row["check"] = "PASS" if ck.returncode == 0 else ("WARN" if "WARN" in ck.stdout else "FAIL")
    else:
        row["check"] = "NOT_COLLECTED"
    return row


# ------------------------------------------------------------------ main
def inner(a):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lab = Lab()
    lab.up()
    collector = lab.spawn("server", [a.python, "-c",
                          "from ai_fingerprint.result_collection import serve_collector_forever as s; "
                          f"s({{'host': '{ROLES['server'][0]}', 'port': 8090, 'root': '{out / 'collected'}'}})"],
                          out / "collector.log", ROLES["server"][1])
    time.sleep(3)
    runs = sorted(Path(p) for p in glob.glob(os.path.join(a.plan, "runs", "*")))
    if a.only:
        runs = [r for r in runs if any(f"_{o}_" in r.name for o in a.only)]
    res = out / "runs.csv"
    done = {r["run"] for r in csv.DictReader(open(res))} if res.exists() else set()
    try:
        for i, r in enumerate(runs, 1):
            if r.name in done:
                continue
            row = run_one(lab, r, out, a.rounds, a.python, a.proxy_template, a.timeout)
            new = not res.exists()
            with res.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"[{i}/{len(runs)}] {row['check']:13s} {row['variant']:34s} {row['seconds_fl']:7.1f}s "
                  f"rc={[row[k] for k in row if k.startswith('rc_')]}", flush=True)
    finally:
        collector.kill()
        lab.down()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="campaign folder from plan_campaign.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--timeout", type=int, default=5400, help="per-run limit for the FL part (s)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--proxy-template",
                    default=str(REPO / "collected_experiments/old/run_20260907t220204z_e367861f/proxy/config.yaml"))
    ap.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.inside:
        return inner(a)
    cmd = ["unshare", "--user", "--map-root-user", "--net", "--fork", "--kill-child",
           sys.executable, __file__, "--inside"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
