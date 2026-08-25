# v0.8.5 experiment-5 checklist

Use one new shared experiment ID on the server, both clients, and proxy.

## 1. Server

In the interactive workflow choose:

```text
task:       training
deployment: federated
role:       server
transport:  tls
minimum:    TLSv1_2
```

When asked for a TLS certificate path, leave it blank to let the code create a
short-lived self-signed research certificate under:

```text
experiments/results/_tls/
```

The server ground truth will report `transport=tls`.

## 2. Clients

Choose the same model/dataset/round configuration and:

```text
transport: tls
verify server certificate: no
```

for the first research collection unless a shared CA/certificate has already
been installed.

Each federated client now writes one `network_registration` event containing:

```text
client_id
local_ip
local_port
remote_host
remote_port
transport
```

The mapping is ground-truth metadata only.

## 3. Proxy

Keep the offload lifecycle enabled and required.

Enter the participating client IPs for capture isolation. For capture aliases,
the recommended choice is to leave the alias prompt blank. The proxy will use
neutral names:

```text
trace_001
trace_002
```

The actual FL client IDs are resolved later by IP from the client
network-registration events. This prevents accidental manual alias reversal.

## 4. Ground-truth phase semantics

New client logs use:

```text
Download
Training
Upload
Idle
```

Upload is only the model-send interval.

Idle with:

```text
reason=synchronous_round_wait
```

is the post-upload synchronous wait.

The diagnostic `federated_upload_transaction` record includes both intervals
and the total transaction time.

## 5. Timing alignment

Per-client feature CSVs contain metadata fields:

```text
trace_start_offset_sec
trace_end_offset_sec
window_start_global_sec
window_end_global_sec
```

They are relative to the combined proxy capture and are excluded from model X.

All ground-truth events also contain:

```text
timestamp_utc
timestamp_epoch
timestamp_monotonic_ns
```

Use UTC/epoch for cross-host alignment. Monotonic timestamps are host-local.

## 6. Resource telemetry

The resource files now record:

```text
actual_interval_ms
sampling_duration_ms
sampling_overrun_ms
```

A system with `nvidia-smi` installed but no usable NVIDIA GPU will no longer
invoke the failed GPU query every sample.

## 7. Dataset preparation

After copying proxy, client, and server outputs under one project directory,
run:

```bash
python3 prepare_fingerprinting_dataset.py
```

The script will print the automatically resolved mapping, for example:

```text
EXP_005: 10.42.0.210 trace_001 -> client_1
EXP_005: 10.42.0.47  trace_002 -> client_2
```

The IP mapping, resolved client ID, and global timing metadata remain
non-predictor metadata.
