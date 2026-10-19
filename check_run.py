#!/usr/bin/env python3
"""
Check collected FL runs right after collection, so an incomplete run is repeated immediately.

For each run directory in collected_experiments it checks:
  roles        server, client_1..3 and proxy folders exist and each status is COMPLETED
  ground truth every client logged one row per planned round (client round_metrics.csv)
  capture      the proxy marked the capture complete and produced one trace per client
  rounds       rounds rebuilt from the proxy's packet sequence alone (extract_round_features)
               cover the planned rounds for every client
  payload      the rebuilt upload size matches the logged model size (within --payload-tol)
  devices      client device labels match the campaign plan (with --plan)
  drops        the proxy capture dropped no packets (FAIL above --max-drop-frac, WARN for any)

A run is FAIL if any required check fails, WARN if only soft checks do (e.g. a few rounds missing),
PASS otherwise. The exit code is 1 if any run FAILs.

Usage
  python3 check_run.py collected_experiments/run_20261020t101500z_ab12cd34
  python3 check_run.py collected_experiments --latest 3 --plan campaigns/c2/schedule.csv
"""
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_round_features import load_trace, rounds_for_trace, trace_ips, client_map

CLIENTS = ("client_1", "client_2", "client_3")


def status_of(role_dir):
    p = os.path.join(role_dir, "experiment_status.json")
    if not os.path.exists(p):
        return None
    try:
        return str(json.load(open(p)).get("status"))
    except Exception:
        return "unreadable"


def proxy_capture_complete(run_dir):
    for p in glob.glob(os.path.join(run_dir, "proxy", "*_proxy_summary.json")):
        try:
            cap = json.load(open(p)).get("capture") or {}
            return bool(cap.get("capture_complete"))
        except Exception:
            return False
    return False


def capture_drops(run_dir):
    """Capture statistics the proxy recorded (dumpcap received/dropped), or None for runs
    collected before the proxy saved them."""
    p = os.path.join(run_dir, "proxy", "capture_chunks_manifest.json")
    if not os.path.exists(p):
        return None
    try:
        stats = json.load(open(p)).get("capture_statistics") or {}
    except Exception:
        return None
    return stats if stats.get("packets_dropped") is not None else None


def planned_devices(plan_csv):
    if not plan_csv:
        return None
    pj = os.path.join(os.path.dirname(plan_csv), "plan.json")
    if not os.path.exists(pj):
        return None
    dev = json.load(open(pj)).get("devices", {})
    return {r: d.get("label") for r, d in dev.items()}


def check(run_dir, a, plan_dev):
    res = []   # (check, level, ok, detail); level "fail" = required, "warn" = soft

    def add(name, level, ok, detail=""):
        res.append((name, level, bool(ok), detail))

    run_id = os.path.basename(run_dir.rstrip("/"))
    scfg_p = os.path.join(run_dir, "server", "config.yaml")
    if not os.path.exists(scfg_p):
        add("roles", "fail", False, "server/config.yaml missing")
        return run_id, {}, res
    scfg = yaml.safe_load(open(scfg_p))
    R = int(scfg["federated"]["rounds"])
    ai = scfg["ai"]
    info = {"variant": ai["variant"], "dataset": ai["dataset"], "rounds": R,
            "experiment": scfg["experiment"].get("experiment_id")}

    for role in ("server",) + CLIENTS + ("proxy",):
        st = status_of(os.path.join(run_dir, role))
        add(f"status {role}", "fail", st == "COMPLETED", st or "missing")

    for c in CLIENTS:
        rm = os.path.join(run_dir, c, "round_metrics.csv")
        n = len(pd.read_csv(rm, usecols=["round"])) if os.path.exists(rm) else 0
        add(f"ground truth {c}", "fail", n >= R, f"{n}/{R} rounds logged")
        if plan_dev:
            cc = os.path.join(run_dir, c, "config.yaml")
            got = yaml.safe_load(open(cc))["device"]["label"] if os.path.exists(cc) else None
            add(f"device {c}", "fail", got == plan_dev.get(c), f"{got} (planned {plan_dev.get(c)})")

    add("proxy capture_complete", "fail", proxy_capture_complete(run_dir))
    drops = capture_drops(run_dir)
    if drops is not None:
        n, frac = int(drops["packets_dropped"]), float(drops.get("drop_fraction") or 0.0)
        detail = f"{n} packets dropped ({100 * frac:.4f}%)"
        add("capture drops", "fail", frac <= a.max_drop_frac, detail)
        if n and frac <= a.max_drop_frac:
            add("capture drops (any)", "warn", False, detail)

    seqs = glob.glob(os.path.join(run_dir, "proxy", "**", "*__trace_*_fingerprint_sequence.csv"), recursive=True)
    by_trace = {}
    for p in seqs:
        by_trace.setdefault(re.search(r"__(trace_\d+)_fingerprint_sequence", p).group(1), []).append(p)
    ipmap, cmap = trace_ips(run_dir), client_map(run_dir)
    # a client can open several connections (FLARE: the client daemon and a separate job process);
    # judge each client by the trace that carries its model transfers
    best = {}
    for tid, paths in sorted(by_trace.items()):
        cid, _, _, gt_payload = cmap.get(ipmap.get(tid), (None, None, np.nan, np.nan))
        if cid is None:
            continue
        try:
            Rr = rounds_for_trace(load_trace(paths))
        except Exception as e:
            best.setdefault(cid, (None, gt_payload, f"{type(e).__name__}: {e}"))
            continue
        if best.get(cid, (None,))[0] is None or len(Rr) > len(best[cid][0]):
            best[cid] = (Rr, gt_payload, None)
    found = {}
    for cid, (Rr, gt_payload, error) in sorted(best.items()):
        if Rr is None:
            add(f"rounds {cid}", "fail", False, error)
            continue
        found[cid] = len(Rr)
        missing = R - len(Rr)
        add(f"rounds {cid}", "fail", len(Rr) >= R * a.min_round_frac, f"{len(Rr)}/{R} rebuilt from traffic")
        if 0 < missing and len(Rr) >= R * a.min_round_frac:
            add(f"all rounds {cid}", "warn", False, f"{missing} rounds not rebuilt")
        if len(Rr) and np.isfinite(gt_payload) and gt_payload > 0:
            diff = abs(Rr["up_bytes"].median() - gt_payload)
            err = diff / gt_payload
            # Each upload also carries the protocol's message header and TLS framing (~2 KB,
            # fixed), so allow --payload-overhead bytes on top of the relative tolerance.
            ok = diff <= a.payload_tol * gt_payload + a.payload_overhead
            add(f"payload {cid}", "warn", ok, f"{100 * err:.3f}% (+{int(diff)} B) from logged model size")
    for c in CLIENTS:
        if c not in found:
            add(f"trace {c}", "fail", False, "no proxy trace mapped to this client")
    return run_id, info, res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="run directories, or a collected_experiments directory")
    ap.add_argument("--latest", type=int, help="with a collected_experiments directory: check the N newest runs")
    ap.add_argument("--plan", help="campaign schedule.csv (its plan.json supplies the planned devices)")
    ap.add_argument("--min-round-frac", type=float, default=0.95,
                    help="FAIL if fewer than this share of planned rounds is rebuilt for a client")
    ap.add_argument("--payload-tol", type=float, default=0.01)
    ap.add_argument("--payload-overhead", type=int, default=8192,
                    help="fixed bytes per upload allowed for message header and TLS framing")
    ap.add_argument("--max-drop-frac", type=float, default=0.001,
                    help="FAIL if the proxy capture dropped more than this share of packets (WARN for any drop)")
    ap.add_argument("--report", help="append one row per check to this CSV")
    a = ap.parse_args()

    runs = []
    for p in a.paths:
        p = p.rstrip("/")
        if os.path.basename(p).startswith("run_"):
            runs.append(p)
        else:
            rs = sorted((q for q in glob.glob(os.path.join(p, "run_*")) if os.path.isdir(q)), key=os.path.getmtime)
            runs += rs[-a.latest:] if a.latest else rs
    plan_dev = planned_devices(a.plan)

    rows, failed = [], 0
    for run_dir in runs:
        run_id, info, res = check(run_dir, a, plan_dev)
        fail = any(lvl == "fail" and not ok for _, lvl, ok, _ in res)
        warn = any(lvl == "warn" and not ok for _, lvl, ok, _ in res)
        verdict = "FAIL" if fail else "WARN" if warn else "PASS"
        failed += fail
        print(f"\n{verdict}  {run_id}  {info.get('variant', '?')} / {info.get('dataset', '?')} "
              f"{info.get('experiment', '')}  ({info.get('rounds', '?')} rounds)")
        for name, lvl, ok, detail in res:
            if not ok or a.report is None:
                mark = "ok  " if ok else ("FAIL" if lvl == "fail" else "warn")
                print(f"   {mark}  {name:24s} {detail}")
            rows.append({"run_id": run_id, **info, "check": name, "level": lvl, "ok": ok,
                         "detail": detail, "verdict": verdict})
    if a.report and rows:
        df = pd.DataFrame(rows)
        df.to_csv(a.report, mode="a", index=False, header=not os.path.exists(a.report))
    print(f"\n{len(runs)} run(s) checked, {failed} FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
