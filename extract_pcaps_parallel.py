#!/usr/bin/env python3
"""
Extract a run's proxy capture on the DGX, one pcap chunk per worker process.

The proxy normally extracts its rotated pcap chunks itself, sequentially, after the run ends
(about 16-18 min per 2 GB chunk on the proxy laptop, plus roughly as much disk again for the
CSVs). With `capture.extract_after: false` in the proxy config, the proxy only captures; copy
its pcap chunks to the DGX and run this script. It calls the same extract_capture_artifacts()
with the same settings the proxy would use, so the outputs are identical in name and format
(collected_experiments/<run>/proxy/capture_artifacts/chunk_NNNN/...) and every downstream
script (canonicalize_central_proxy.py, extract_round_features.py, check_run.py) works unchanged.

Per-client traces are split by the client connections the proxy discovered during the run,
read from the collected proxy summary (discovered_client_connections), so trace_001, trace_002,
... mean the same connection in every chunk. If the summary is missing, connections are
recovered from the TCP SYNs to the proxy port in the pcaps, in order of first appearance.

Usage
  # 1. copy the pcap chunks from the proxy laptop (pcaps are never part of result collection)
  rsync -av --progress <user>@<proxy-host>:<repo>/experiments/staging/<run_id>/proxy/pcap/ \\
        /data/pcaps/<run_id>/
  # 2. extract in parallel into the collected run
  python3 extract_pcaps_parallel.py collected_experiments/<run_id> --pcap-dir /data/pcaps/<run_id>
  # 3. verify
  python3 check_run.py collected_experiments/<run_id>
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_fingerprint.traffic.analysis import extract_capture_artifacts


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def proxy_settings(run_dir):
    cfg = yaml.safe_load(open(Path(run_dir) / "proxy" / "config.yaml", encoding="utf-8"))
    summary = {}
    for p in (Path(run_dir) / "proxy").glob("*_proxy_summary.json"):
        summary = json.load(open(p))
        break
    return cfg, summary


def chunk_files(pcap_dir):
    """Rotated chunks in capture order. dumpcap ring-buffer names carry the sequence number,
    so name order is the capture order and survives copying (unlike modification times)."""
    files = [p for p in Path(pcap_dir).iterdir()
             if p.is_file() and p.suffix.lower() in (".pcap", ".pcapng") and p.stat().st_size > 0]
    return sorted(files, key=lambda p: p.name)


def connections_from_syn(chunks, proxy_ip, proxy_port, upstream):
    """(ip, port) of every client connection to the proxy, in order of first SYN."""
    seen, out = set(), []
    for c in chunks:
        flt = f"tcp.flags.syn==1 && tcp.flags.ack==0 && tcp.dstport=={proxy_port}"
        if proxy_ip:
            flt += f" && ip.dst=={proxy_ip}"
        res = subprocess.run(["tshark", "-r", str(c), "-Y", flt, "-T", "fields", "-e", "ip.src", "-e", "tcp.srcport"],
                             capture_output=True, text=True, check=False)
        for line in res.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 2 or not parts[1].isdigit() or parts[0] == upstream:
                continue
            key = (parts[0], int(parts[1]))
            if key not in seen:
                seen.add(key)
                out.append({"client_ip": key[0], "client_port": key[1], "connection_key": f"{key[0]}:{key[1]}",
                            "alias": f"trace_{len(out) + 1:03d}", "source": "offline_syn"})
    return out


def extract_one(job):
    t0 = time.time()
    art = extract_capture_artifacts(**job["kwargs"])
    return job["index"], art, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="collected_experiments/<run_id>")
    ap.add_argument("--pcap-dir", required=True, help="directory holding this run's pcap chunks")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel chunks (default: min(chunks, 8)); each worker needs ~4-6 GB RAM per 2 GB chunk")
    ap.add_argument("--force", action="store_true", help="re-extract chunks that already have artifacts")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    run_id = run_dir.name
    cfg, summary = proxy_settings(run_dir)
    cap, prx = cfg.get("capture", {}), cfg.get("proxy", {})
    chunks = chunk_files(a.pcap_dir)
    if not chunks:
        sys.exit(f"no pcap chunks in {a.pcap_dir}")

    proxy_ip = str(prx.get("listen_host", "")) or None
    proxy_port = int(prx.get("listen_port", 8080))
    upstream = str(prx.get("upstream_host", ""))
    conns = summary.get("discovered_client_connections") or []
    ips = summary.get("discovered_client_ips") or []
    source = "proxy_summary"
    if not conns:
        conns = connections_from_syn(chunks, proxy_ip, proxy_port, upstream)
        ips = sorted({c["client_ip"] for c in conns})
        source = "pcap_syn"
    if not conns:
        sys.exit("no client connections found in the proxy summary or the pcaps")
    aliases = {c["connection_key"]: c["alias"] for c in conns}
    print(f"{run_id}: {len(chunks)} chunks, {len(conns)} client connections ({source})", flush=True)

    out_root = run_dir / "proxy" / "capture_artifacts"
    jobs = []
    for i, chunk in enumerate(chunks, 1):
        cdir = out_root / f"chunk_{i:04d}"
        if not a.force and any(cdir.glob("*_manifest.json")):
            print(f"  chunk {i:04d}: already extracted, skipped", flush=True)
            continue
        jobs.append({"index": i, "kwargs": dict(
            pcap_path=str(chunk), experiment_id=run_id, output_dir=str(cdir),
            server_ip=cap.get("proxy_client_facing_ip") or None, client_ip=cap.get("client_ip") or None,
            client_ips=ips, client_aliases=aliases, client_connections=conns,
            proxy_ip=proxy_ip, proxy_port=proxy_port,
            per_client_artifacts=bool(cap.get("per_client_artifacts", True)),
            capture_interface=str(cap.get("interface") or ""),
            burst_gap_sec=float(cap.get("burst_gap_sec", 0.05)),
            idle_threshold_sec=float(cap.get("idle_threshold_sec", 0.5)),
            window_seconds=cap.get("window_seconds"), window_sizes_sec=cap.get("window_sizes_sec") or None,
            capture_isolation_metadata={"mode": "offline_parallel_extraction", "connection_source": source,
                                        "chunk_index": i, "chunk_count": len(chunks),
                                        "packet_information_threshold": int(cap.get("packet_information_threshold", 2)),
                                        "upstream_server_excluded": upstream},
            packet_information_threshold=int(cap.get("packet_information_threshold", 2)))})

    workers = a.workers or min(len(jobs), 8) or 1
    results, failed = {}, []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(extract_one, j): j["index"] for j in jobs}
        for f in as_completed(futs):
            i = futs[f]
            try:
                _, art, sec = f.result()
                results[i] = art
                print(f"  chunk {i:04d}: {art.get('packet_count', 0):>10,} packets in {sec / 60:5.1f} min", flush=True)
            except Exception as e:
                failed.append(i)
                print(f"  chunk {i:04d}: FAILED {type(e).__name__}: {e}", flush=True)

    # merge with chunks extracted earlier, then write the index the proxy would have written
    for i in range(1, len(chunks) + 1):
        if i not in results:
            m = next((out_root / f"chunk_{i:04d}").glob("*_manifest.json"), None)
            if m is not None:
                results[i] = json.load(open(m))
    per_client = {}
    for i in sorted(results):
        per_client.update(results[i].get("per_client_artifacts", {}) or {})
    index = {"experiment_id": run_id, "pcap_chunks": len(chunks),
             "total_packets": sum(int(r.get("packet_count", 0) or 0) for r in results.values()),
             "feature_row_count": sum(int(r.get("feature_row_count", 0) or 0) for r in results.values()),
             "chunk_artifacts": [results[i] for i in sorted(results)], "per_client_artifacts": per_client,
             "streaming_extraction": "offline_parallel", "connection_source": source,
             "chunks_failed": failed, "capture_complete": not failed and len(results) == len(chunks),
             "workers": workers, "wall_time_sec": round(time.time() - t_start, 1), "timestamp_utc": utc_now()}
    out_root.parent.joinpath("capture_artifacts_index.json").write_text(json.dumps(index, indent=2, default=str))
    print(f"{run_id}: {len(results)}/{len(chunks)} chunks extracted in {(time.time() - t_start) / 60:.1f} min "
          f"with {workers} workers; {len(failed)} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
