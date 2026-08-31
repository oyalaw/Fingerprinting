# Central Result Collection and Fingerprinting

## Scientific separation

The experiment node that performs live proxy observation remains label blind. The default central collector is the FL server (or another analysis-only host), not the proxy. The collector receives experiment copies for post-run analysis only.

Local participant results are authoritative originals. Central files are verified replicas. A collection failure never changes a valid experiment from `COMPLETED` to `FAILED`.

## One-time collector startup

On the central Ubuntu analysis host, from the repository root:

```bash
python result_collector.py
```

Default endpoint: `0.0.0.0:8090`.
Default storage root: `collected_experiments/`.
Copy `collector.example.yaml` to `collector.yaml` to override the port/root or configure a shared token.

## Automatic participant replication

For federated client/server interactive runs, answer yes to centralized result replication and provide the collector IP/hostname. The default port is `8090`.

At terminal experiment completion each participant:

1. retains its original local output unchanged;
2. selects only analysis-relevant CSV/JSON/JSONL/YAML/TXT/log files;
3. excludes raw PCAP, checkpoints, TLS keys, and model binaries;
4. creates per-file SHA256 hashes and a collection manifest;
5. uploads a temporary ZIP copy;
6. the collector verifies archive SHA256 plus every file SHA256;
7. the participant writes `collection_receipt.json` only after verification.

If the collector is unavailable, `collection_status.json` becomes `PENDING` and the local result remains valid.

Retry later from the participant repository root:

```bash
python collect_pending_results.py
```

## Central layout

```text
collected_experiments/
  <run_id>/
    client_1/
    client_2/
    client_3/
    server/
    proxy/
    _archives/
    collection_index.json
```

The proxy uploads only post-capture analysis artifacts. Raw PCAP stays local on the proxy.

## Central fingerprinting

After the required participant copies have arrived:

```bash
python run_central_fingerprinting.py
```

The command validates every collected run, excludes partial/incomplete runs, builds `fingerprinting_dataset/` using proxy-observable network features only, and runs hierarchical fingerprint training:

`family -> architecture -> variant -> application`.

Publication evaluation remains grouped by independent experiment/run ID; windows from one run must never be split randomly across train and test.
