# v0.8.3 migration notes

## What changed

Version 0.8.3 hardens the proxy capture path for multi-client fingerprinting.
The proxy no longer permits an unscoped capture by default. Participating
client IPs are required so only the client-facing leg is captured.

## Recommended proxy setup

Run:

```bash
python3 main.py
```

Choose `proxy`. When prompted, provide every participating client IP. For a
federated run, also map those IPs to the exact `client_id` values used by the
clients, for example:

```text
Participating client IPs: 10.42.0.47,10.42.0.210
Optional client aliases: 10.42.0.47=client_1,10.42.0.210=client_2
```

The resulting BPF is equivalent to:

```text
(host 10.42.0.47 or host 10.42.0.210) and port <proxy-listen-port>
```

The upstream server is therefore excluded from packet capture even if the
proxy-to-server connection uses the same physical interface and TCP port.

## Storage defaults

The default PCAP snapshot length is 256 bytes. The original frame length is
still retained by pcap/pcapng and is used by `frame.len`, while most encrypted
payload bytes are not stored. Enter `0` at the prompt if full frames are
required for a specific experiment.

The per-recv forwarding diagnostic CSV is disabled by default. Aggregate
forwarded byte and connection counters remain in the proxy summary.

## Windowing

The proxy writes one overall feature row plus 5-second windows by default.
Change the value at the prompt or enter `0` for overall-only extraction.

## Multi-client outputs

For a capture with aliases `client_1` and `client_2`, the proxy writes, among
other files:

```text
EXP_packet_sequence.csv
EXP_fingerprint_sequence.csv
EXP_features.csv
EXP_manifest.json
EXP__client_1_fingerprint_sequence.csv
EXP__client_1_features.csv
EXP__client_2_fingerprint_sequence.csv
EXP__client_2_features.csv
```

For classifier construction, use the per-client files. The combined
multi-client trace is retained for audit/aggregate analysis and is marked
classifier-ineligible in the manifest.

Federated client local files now also include the client ID, for example:

```text
EXP_client_1_ground_truth.jsonl
EXP_client_1_resource.csv
EXP_client_1_resource_summary.json
```

This avoids filename collisions when logs from several client devices are
copied into one analysis directory.

## Repairing an earlier broad capture without its PCAP

If the raw packet sequence still exists, run:

```bash
python3 repair_proxy_sequence.py
```

Select the raw `*_packet_sequence.csv`, enter all participating client IPs,
and map them to their federated client IDs when applicable. The utility:

1. removes packets that do not involve those client IPs;
2. eliminates the proxy-to-upstream duplicate leg;
3. recomputes up/down direction from client membership;
4. reconstructs SYN/ACK/FIN/RST from `tcp_flags_hex`;
5. writes a cleaned raw sequence and classifier-safe sequence;
6. produces separate per-client features/sequences;
7. creates an overall row plus configured time windows.

The original PCAP is not required for this repair path.

## Capture-quality warnings

The proxy records interface MTU and, when `ethtool` is installed, the state of
GRO, GSO, TSO, and LRO. It also reports captured frames substantially above the
interface MTU as possible host offload/coalescing artifacts. The software does
not change network-interface settings automatically.
