#!/usr/bin/env python3
"""
One-time validation of the proxy capture against a capture taken on a client (pilot only).

During one pilot run, capture headers only on ONE client (see docs/validation_capture.md), then
compare that client's view with the proxy's trace of the same client:

  * rounds       FL rounds rebuilt from each capture (same reconstruction as the dataset)
  * bytes        per-round download and upload bytes seen by the client vs the proxy
  * timing       offset between the two clocks and its spread across rounds (proxy - client),
                 from the start of each upload; a stable offset means the proxy timing is faithful
  * coverage     share of the client's payload bytes that the proxy also saw

Packet counts are not compared: the client's network stack may coalesce segments (TSO/GRO) in
its own capture, while the proxy disables offloads. Bytes and round timing are what matter.

Usage
  python3 compare_client_capture.py --client-pcap client1_check.pcapng --client-ip 10.42.0.210 \\
      --run collected_experiments/<run_id> --client-id client_1
"""
import argparse, glob, os, re, subprocess, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_round_features import client_map, load_trace, rounds_for_trace, trace_header_length, trace_ips


def client_side_trace(pcap, client_ip, server_port):
    """Packets of the client's FL connection as seen on the client: time, payload, direction."""
    out = subprocess.run(
        ["tshark", "-r", pcap, "-Y", f"tcp && ip.addr=={client_ip} && tcp.port=={server_port}",
         "-T", "fields", "-e", "frame.time_epoch", "-e", "frame.len", "-e", "ip.src"],
        capture_output=True, text=True, check=True).stdout
    rows = [l.split("\t") for l in out.splitlines() if l.count("\t") == 2]
    d = pd.DataFrame(rows, columns=["t", "frame_length", "src"])
    d["relative_time_sec"] = d["t"].astype(float)
    d["frame_length"] = d["frame_length"].astype(float)
    d["direction"] = np.where(d["src"] == client_ip, "up", "down")
    d = d.sort_values("relative_time_sec").reset_index(drop=True)
    d["payload"] = (d["frame_length"] - trace_header_length(d)).clip(lower=0).astype("int64")
    return d[d["payload"] > 0][["relative_time_sec", "payload", "direction"]].reset_index(drop=True)


def proxy_trace(run_dir, client_id):
    seqs = glob.glob(os.path.join(run_dir, "proxy", "**", "*__trace_*_fingerprint_sequence.csv"), recursive=True)
    by_trace = {}
    for p in seqs:
        by_trace.setdefault(re.search(r"__(trace_\d+)_fingerprint_sequence", p).group(1), []).append(p)
    ips, cmap = trace_ips(run_dir), client_map(run_dir)
    for tid, paths in by_trace.items():
        if cmap.get(ips.get(tid), (None,))[0] == client_id:
            return load_trace(paths), tid
    sys.exit(f"no proxy trace for {client_id} in {run_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--client-pcap", required=True)
    ap.add_argument("--client-ip", required=True)
    ap.add_argument("--server-port", type=int, default=8080, help="the port the client connects to (proxy port)")
    ap.add_argument("--run", required=True, help="collected_experiments/<run_id>")
    ap.add_argument("--client-id", required=True)
    a = ap.parse_args()

    c = client_side_trace(a.client_pcap, a.client_ip, a.server_port)
    p, tid = proxy_trace(a.run, a.client_id)
    rc, rp = rounds_for_trace(c), rounds_for_trace(p)
    print(f"{a.client_id}: client capture {len(c):,} data packets, proxy {tid} {len(p):,} data packets")
    print(f"rounds rebuilt: client view {len(rc)}, proxy view {len(rp)}")
    n = min(len(rc), len(rp))
    if n == 0:
        sys.exit("no rounds rebuilt from one of the captures")
    # proxy times are relative to the capture start; align on the first round, then measure drift
    rc, rp = rc.iloc[:n].reset_index(drop=True), rp.iloc[:n].reset_index(drop=True)
    shift = rp["up_t0"].iloc[0] - rc["up_t0"].iloc[0]
    resid = (rp["up_t0"] - rc["up_t0"] - shift) * 1000.0
    for col in ("down_bytes", "up_bytes"):
        rel = (rp[col] - rc[col]).abs() / rc[col].replace(0, np.nan)
        print(f"{col:11s} median relative difference {100 * rel.median():.4f}%  (max {100 * rel.max():.4f}%)")
    gap = (rp["compute_gap"] - rc["compute_gap"]) * 1000.0
    print(f"upload-start timing: residual after alignment median {resid.median():.2f} ms, "
          f"IQR {resid.quantile(.75) - resid.quantile(.25):.2f} ms, max |{resid.abs().max():.2f}| ms")
    print(f"compute gap, proxy - client: median {gap.median():.2f} ms, IQR {gap.quantile(.75) - gap.quantile(.25):.2f} ms")
    print(f"payload coverage: proxy saw {100 * p['payload'].sum() / c['payload'].sum():.3f}% of the client's bytes")


if __name__ == "__main__":
    main()
