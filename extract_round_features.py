#!/usr/bin/env python3
"""
Per-round passive features from the proxy's per-client packet sequences.

For every run in collected_experiments that has proxy/**/<run>__trace_XXX_fingerprint_sequence.csv,
each FL round of each client is reconstructed from direction + timing only (what a passive
on-path observer sees):

    download transfer (server->client model)  ->  compute gap  ->  upload transfer (client->server)

Outputs (small; upload these):
    round_features.csv   one row per client per round
    trace_summary.csv    one row per client trace (medians etc.), joined to client id/device

Usage (on the DGX, from the Fingerprinting repo root):
    python3 extract_round_features.py collected_experiments --out-dir round_features
    # optional: --runs run_metadata.csv  (only runs listed there with a family label)
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
import pandas as pd

HDR = 66            # Ethernet(14) + IPv4(20) + TCP with timestamps(32); connections without the
                    # timestamp option use 54 (see trace_header_length)
MERGE_GAP = 900.0   # s. Rounds strictly alternate download/upload, so transfers are split on direction
                    # change only; silence inside a transfer (Wi-Fi stalls, TCP RTO backoff of 10-100 s on
                    # large models) must NOT split it. 900 s only guards against joining across restarts.
CTRL_MAX = 300      # opposite-direction packets with payload <= this do not break a transfer (control msgs)


def chunk_start_epoch(path):
    """capture_start_epoch of the chunk a sequence file belongs to (from the chunk manifest)."""
    for m in glob.glob(os.path.join(os.path.dirname(path), "*_manifest.json")):
        try:
            j = json.load(open(m))
            e = (j.get("capture") or {}).get("capture_start_epoch")
            if e:
                return float(e)
        except Exception:
            pass
    return None


def load_trace(paths):
    """Load one client's packets across all capture chunks onto one timeline.
    Long runs are split into chunks whose relative_time_sec restarts near 0; each chunk is shifted
    by (its capture_start_epoch - the first chunk's) unless its times are already global."""
    cols = ["relative_time_sec", "frame_length", "direction", "tcp_len"]
    starts = {p: chunk_start_epoch(p) for p in paths}
    known = [e for e in starts.values() if e is not None]
    t_ref = min(known) if known else None
    parts = []
    for p in sorted(paths, key=lambda q: (starts[q] or 0, q)):
        # tolerant read: truncated last rows / NA frame lengths are dropped
        d = pd.read_csv(p, usecols=lambda c: c in cols,
                        dtype={"frame_length": "float64", "direction": "string"}, on_bad_lines="skip")
        if "tcp_len" not in d:
            d["tcp_len"] = -1.0
        d = d.dropna(subset=["relative_time_sec", "frame_length", "direction"])
        if len(d) and t_ref is not None and starts[p] is not None:
            off = starts[p] - t_ref
            if off > 1 and d["relative_time_sec"].min() < off - 30:   # chunk-local clock -> shift
                d["relative_time_sec"] += off
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d.sort_values("relative_time_sec", kind="stable").reset_index(drop=True)
    # Exact TCP payload (tshark tcp.len) where the sequence has it; otherwise frame length minus
    # the connection's inferred header length.
    estimate = (d["frame_length"] - trace_header_length(d)).clip(lower=0)
    exact = pd.to_numeric(d["tcp_len"], errors="coerce").fillna(-1)
    d["payload"] = exact.where(exact >= 0, estimate).astype("int64")
    return d[d["payload"] > 0][["relative_time_sec", "payload", "direction"]].reset_index(drop=True)



def trace_header_length(d):
    """Header bytes per frame for this connection, inferred from pure acknowledgments: 66 with the
    TCP timestamp option (Linux default), 54 without it (Windows default). Assuming 66 for a
    connection without timestamps under-counts every full data packet by 12 bytes (~0.8%)."""
    small = d["frame_length"].value_counts()
    return 54 if small.get(54, 0) > small.get(HDR, 0) else HDR

def fine_transfers(d):
    """Pass 1: same-direction runs of data packets, tolerating tiny opposite-direction packets."""
    t = d["relative_time_sec"].to_numpy()
    pl = d["payload"].to_numpy()
    dr = (d["direction"].astype(str).to_numpy() == "up")
    out = []
    cur = None
    for i in range(len(t)):
        if cur is not None and dr[i] != cur["up"] and pl[i] <= CTRL_MAX and t[i] - cur["t1"] < MERGE_GAP:
            cur["ctrl_bytes"] += int(pl[i])
            continue
        if cur is None or dr[i] != cur["up"] or t[i] - cur["t1"] >= MERGE_GAP:
            if cur is not None:
                out.append(cur)
            cur = {"up": bool(dr[i]), "t0": float(t[i]), "t1": float(t[i]), "bytes": 0, "pkts": 0, "ctrl_bytes": 0}
        cur["t1"] = float(t[i])
        cur["bytes"] += int(pl[i])
        cur["pkts"] += 1
    if cur is not None:
        out.append(cur)
    return out


def transfers(d, rel=0.3):
    """Pass 2: remove interruptions. A piece is absorbed as noise only when BOTH neighbours go the
    other way (same direction as each other), it is tiny relative to both (<= rel * smaller
    neighbour), and the neighbours are within MERGE_GAP; the two neighbours are then joined.
    Smallest pieces are handled first, and neighbourhoods are re-checked after each merge.
    The start of a real transfer is never absorbed, because its neighbours differ in direction
    or it is not tiny relative to both."""
    import heapq
    P = fine_transfers(d)
    n = len(P)
    if n == 0:
        return pd.DataFrame()
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1)); nxt[-1] = -1
    alive = [True] * n
    heap = [(P[i]["bytes"], i) for i in range(n)]
    heapq.heapify(heap)

    def eligible(i):
        a, b = prev[i], nxt[i]
        if a < 0 or b < 0:
            return False
        pa, pb, pi = P[a], P[b], P[i]
        return (pa["up"] == pb["up"] != pi["up"]
                and pi["bytes"] <= rel * min(pa["bytes"], pb["bytes"])
                and pb["t0"] - pa["t1"] < MERGE_GAP)

    while heap:
        sz, i = heapq.heappop(heap)
        if not alive[i] or sz != P[i]["bytes"] or not eligible(i):
            continue
        a, b = prev[i], nxt[i]
        pa, pb = P[a], P[b]
        pa["ctrl_bytes"] += P[i]["bytes"] + pb["ctrl_bytes"]
        pa["bytes"] += pb["bytes"]; pa["pkts"] += pb["pkts"]; pa["t1"] = pb["t1"]
        alive[i] = alive[b] = False
        c = nxt[b]
        nxt[a] = c
        if c >= 0:
            prev[c] = a
            heapq.heappush(heap, (P[c]["bytes"], c))
        if prev[a] >= 0:
            heapq.heappush(heap, (P[prev[a]]["bytes"], prev[a]))
    return pd.DataFrame([P[i] for i in range(n) if alive[i]])


def model_transfers(T, up):
    """Keep the transfers that carry the model. The reference size is the byte-weighted median
    transfer size (the size class that carries most of the bytes), so thousands of small
    fragments or control messages cannot outvote ~100 model transfers."""
    x = T[T["up"] == up]
    if x.empty:
        return x
    b = np.sort(x["bytes"].to_numpy())
    cum = np.cumsum(b)
    ref = b[np.searchsorted(cum, cum[-1] / 2)]
    return x[(x["bytes"] >= 0.8 * ref) & (x["bytes"] <= 1.2 * ref)]


def rounds_for_trace(d):
    T = transfers(d)
    if T.empty:
        return pd.DataFrame()
    ups = model_transfers(T, True).reset_index(drop=True)
    downs = model_transfers(T, False).reset_index(drop=True)
    rows = []
    T = T.sort_values("t0").reset_index(drop=True)
    down_keys = set(zip(downs["t0"], downs["t1"]))
    skipped = 0
    for k, u in ups.iterrows():
        # the last substantial transfer before this upload must be a model download;
        # otherwise a round was missed and pairing would span several rounds -> skip
        before = T[(T["t1"] <= u["t0"]) & (T["bytes"] >= 0.3 * u["bytes"])]
        if before.empty:
            continue
        dn = before.iloc[-1]
        if dn["up"] or (dn["t0"], dn["t1"]) not in down_keys:
            skipped += 1
            continue
        nxt = downs[downs["t0"] >= u["t1"]]
        prev_up_t0 = ups["t0"].iloc[k - 1] if k > 0 else np.nan
        between = T[(T["t0"] > dn["t1"]) & (T["t1"] < u["t0"])]
        rows.append({
            "round": len(rows),
            "down_t0": dn["t0"], "down_t1": dn["t1"], "down_bytes": dn["bytes"], "down_pkts": dn["pkts"],
            "down_duration": dn["t1"] - dn["t0"],
            "up_t0": u["t0"], "up_t1": u["t1"], "up_bytes": u["bytes"], "up_pkts": u["pkts"],
            "up_duration": u["t1"] - u["t0"],
            "compute_gap": u["t0"] - dn["t1"],
            "gap_msgs": len(between), "gap_msg_bytes": int(between["bytes"].sum()) if len(between) else 0,
            "wait_after_upload": (nxt["t0"].iloc[0] - u["t1"]) if len(nxt) else np.nan,
            "round_period": u["t0"] - prev_up_t0,
        })
    R = pd.DataFrame(rows)
    if not R.empty:
        R["down_throughput_Bps"] = R["down_bytes"] / R["down_duration"].replace(0, np.nan)
        R["up_throughput_Bps"] = R["up_bytes"] / R["up_duration"].replace(0, np.nan)
        R["compute_gap_per_MB"] = R["compute_gap"] / (R["up_bytes"] / 1e6)
    return R


def client_map(run_dir):
    """client ip -> (client_id, device, gt median training time, gt payload)."""
    m = {}
    for cdir in glob.glob(os.path.join(run_dir, "client_*")):
        cid = os.path.basename(cdir)
        ip, device = None, None
        for gt in glob.glob(os.path.join(cdir, "*ground_truth.jsonl")):
            with open(gt) as f:
                for i, line in enumerate(f):
                    if i > 50:
                        break
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if e.get("local_ip"):
                        ip = e["local_ip"]
                    device = device or e.get("device")
                    if ip:
                        break
        gt_train = gt_payload = np.nan
        rm = os.path.join(cdir, "round_metrics.csv")
        if os.path.exists(rm):
            try:
                r = pd.read_csv(rm, usecols=lambda c: c in ("training_time_sec", "model_size_bytes"))
                gt_train = r["training_time_sec"].median() if "training_time_sec" in r else np.nan
                gt_payload = r["model_size_bytes"].median() if "model_size_bytes" in r else np.nan
            except Exception:
                pass
        if ip:
            m[ip] = (cid, device, gt_train, gt_payload)
    return m


def trace_ips(run_dir):
    """trace_00X -> client ip, from the proxy summary / capture index."""
    out = {}
    for p in glob.glob(os.path.join(run_dir, "proxy", "**", "*.json"), recursive=True):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        stack = [j]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                pca = o.get("per_client_artifacts")
                if isinstance(pca, dict):
                    for tid, v in pca.items():
                        if isinstance(v, dict) and v.get("client_ip"):
                            out[tid] = v["client_ip"]
                stack += list(o.values())
            elif isinstance(o, list):
                stack += o
    return out


def main():
    global MERGE_GAP
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="collected_experiments directory (or a single run dir)")
    ap.add_argument("--runs", help="run_metadata.csv; restrict to labeled runs listed there")
    ap.add_argument("--out-dir", default="round_features")
    ap.add_argument("--merge-gap", type=float, default=MERGE_GAP,
                    help="max silence (s) inside one transfer before it is split (default 900)")
    ap.add_argument("--only", nargs="*", help="run id suffixes to process, e.g. fd5f756f a6b65536")
    ap.add_argument("--debug", action="store_true",
                    help="also write per-trace transfer lists + frame-size histograms to <out-dir>/debug/")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    MERGE_GAP = a.merge_gap

    if os.path.basename(a.root.rstrip("/")).startswith("run_"):
        run_dirs = [a.root.rstrip("/")]
    else:
        run_dirs = sorted(p for p in glob.glob(os.path.join(a.root, "run_*")) if os.path.isdir(p))
    if a.runs:
        meta = pd.read_csv(a.runs)
        keep = set(meta.loc[meta["family"].notna(), "experiment_id"])
        run_dirs = [r for r in run_dirs if os.path.basename(r) in keep]

    if a.only:
        run_dirs = [r for r in run_dirs if any(r.endswith(x) for x in a.only)]
    if a.debug:
        os.makedirs(os.path.join(a.out_dir, "debug"), exist_ok=True)

    all_rounds, summaries = [], []
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        seqs = glob.glob(os.path.join(run_dir, "proxy", "**", "*__trace_*_fingerprint_sequence.csv"), recursive=True)
        if not seqs:
            continue
        by_trace = {}
        for p in seqs:
            tid = re.search(r"__(trace_\d+)_fingerprint_sequence", p).group(1)
            by_trace.setdefault(tid, []).append(p)
        ipmap, cmap = trace_ips(run_dir), client_map(run_dir)
        for tid, paths in sorted(by_trace.items()):
            try:
                d = load_trace(paths)
                R = rounds_for_trace(d)
                if a.debug:
                    base = os.path.join(a.out_dir, "debug", f"{run_id[-8:]}_{tid}")
                    fine = pd.DataFrame(fine_transfers(d))
                    fine.head(3000).to_csv(base + "_fine_transfers_first3000.csv", index=False)
                    transfers(d).to_csv(base + "_merged_transfers.csv", index=False)
                    (d.assign(size=d["payload"].clip(upper=1448)).groupby(["direction", "size"]).size()
                       .rename("count").reset_index().sort_values("count", ascending=False).head(200)
                       .to_csv(base + "_payload_histogram.csv", index=False))
            except Exception as e:
                print(f"  {run_id[-8:]} {tid}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
                continue
            ip = ipmap.get(tid)
            cid, dev, gt_train, gt_payload = cmap.get(ip, (None, None, np.nan, np.nan))
            if R.empty:
                print(f"  {run_id[-8:]} {tid}: no rounds found")
                continue
            for k, v in {"experiment_id": run_id, "trace_id": tid, "client_ip": ip,
                         "client_id": cid, "device": dev}.items():
                R.insert(0, k, v)
            all_rounds.append(R)
            q = R["compute_gap"]
            summaries.append({
                "experiment_id": run_id, "trace_id": tid, "client_ip": ip, "client_id": cid, "device": dev,
                "chunks": len(paths), "rounds_found": len(R),
                "payload_up_bytes": int(R["up_bytes"].median()), "payload_down_bytes": int(R["down_bytes"].median()),
                "payload_up_cv": R["up_bytes"].std() / R["up_bytes"].mean(),
                "compute_gap_median": q.median(), "compute_gap_iqr": q.quantile(.75) - q.quantile(.25),
                "compute_gap_p10": q.quantile(.1), "compute_gap_p90": q.quantile(.9),
                "compute_gap_per_MB_median": R["compute_gap_per_MB"].median(),
                "up_throughput_median": R["up_throughput_Bps"].median(),
                "down_throughput_median": R["down_throughput_Bps"].median(),
                "round_period_median": R["round_period"].median(),
                "wait_after_upload_median": R["wait_after_upload"].median(),
                "gap_msgs_median": R["gap_msgs"].median(),
                "gt_payload_bytes": gt_payload, "gt_training_time_median": gt_train,
            })
            s = summaries[-1]
            print(f"  {run_id[-8:]} {tid} {cid or '?':9s} chunks={len(paths)} rounds={len(R):3d} payload={s['payload_up_bytes']:>10d} "
                  f"(gt {gt_payload:.0f})  gap={s['compute_gap_median']:.3f}s (gt train {gt_train:.3f}s)")

    if not all_rounds:
        sys.exit("No traces processed.")
    tag = "_" + "_".join(a.only) if a.only else ""
    pd.concat(all_rounds, ignore_index=True).to_csv(os.path.join(a.out_dir, f"round_features{tag}.csv"), index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(a.out_dir, f"trace_summary{tag}.csv"), index=False)
    print(f"Wrote {a.out_dir}/round_features.csv and {a.out_dir}/trace_summary.csv")


if __name__ == "__main__":
    main()
