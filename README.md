# AI Fingerprinting Experiment Codebase

Version 0.8.4 enforces attacker-data isolation and client-facing capture integrity: fingerprinting predictors come only from proxy-observable network data, client/server logs provide labels only, resource telemetry is prohibited from predictor input, and packet-sequence identity fields are removed from the classifier-safe sequence.

The same repository supports client execution, server execution, proxy capture, real dataset loading, and ground truth logging.

The attacker facing fingerprinting pipeline uses only network observable information. Framework, runtime, family, architecture, application, dataset, and device labels remain in local client and server ground truth logs and are intentionally excluded from the proxy manifest.











## Capture-interface offload control (v0.8.4)

Packet size is a fingerprinting observable, so capture-side GRO/GSO/TSO/LRO
must not silently coalesce or segment packets differently from the wire view.

The proxy now performs this lifecycle automatically:

```text
read original ethtool state
        ↓
disable enabled mutable GRO/GSO/TSO/LRO
        ↓
verify disabled state
        ↓
start PCAP capture
        ↓
stop PCAP capture
        ↓
restore only settings changed by this run
        ↓
verify restoration
```

The default configuration is:

```yaml
capture:
  offload_management:
    enabled: true
    required: true
    allow_sudo_noninteractive: true
    restore_on_exit: true
    features: [gro, gso, tso, lro]
```

When `required: true`, the proxy fails closed rather than collecting packet-size
fingerprints when `ethtool` is missing, an enabled feature is driver-fixed, a
permission error cannot be resolved, or the disabled state cannot be verified.

Do not run the whole application with `sudo`. The project includes a narrow
sudoers installer that permits only `ethtool -K` for GRO/GSO/TSO/LRO on one
specified interface:

```bash
sudo apt install ethtool
sudo bash scripts/install_offload_sudoers.sh <capture-interface>
```

See `OFFLOAD_PRIVILEGES.md` for the privilege model.

Each proxy run writes:

```text
<experiment_id>_proxy_offload_state.json
```

and also embeds the before/capture/restore report in proxy/manifest metadata.
These values are provenance and quality-control metadata, never classifier
predictors.

## Client-facing capture integrity (v0.8.3)

The inline proxy now requires participating client IPs when capture is enabled.
The capture BPF is built from those client IPs plus the proxy listen port, so a
proxy-to-upstream copy of the same byte stream is not captured a second time.
For example:

```text
(host 10.42.0.47 or host 10.42.0.210) and port 8080
```

For federated experiments, optional `client_aliases` should map each client IP
to its exact federated `client_id`:

```yaml
capture:
  client_ips:
    - 10.42.0.47
    - 10.42.0.210
  client_aliases:
    10.42.0.47: client_1
    10.42.0.210: client_2
```

The proxy retains a combined raw trace for audit and automatically writes
separate classifier-safe sequence and feature files per client. In a
multi-client run, the mixed combined trace is marked classifier-ineligible;
use the per-client files for model training/evaluation.

Feature extraction now includes an overall row plus 5-second windows by
default. Change `capture.window_seconds` to another positive duration or set it
to `null` for overall-only features.

PCAP capture now defaults to `capture.snaplen_bytes: 256`. Only the first 256
bytes of each frame are stored, while the PCAP preserves the original frame
length used by packet-size features. This can reduce multi-gigabyte captures
substantially. Set the value to `0` when full encrypted frame payloads are
required; a short snap length can reduce deep TLS reassembly completeness.

The proxy's per-`recv()` forwarding CSV is disabled by default with
`proxy.forwarding_log_enabled: false`. Aggregate forwarded bytes and connection
statistics remain in the proxy summary. Enable the CSV only for proxy debugging;
its chunk boundaries are not packet boundaries and it is not fingerprinting
input.

TCP SYN/ACK/FIN/RST fields are reconstructed from the `tcp.flags` bitmask,
which avoids tshark-version cases where individual flag subfields are blank or
zero. The manifest records interface MTU and the v0.8.4 offload-management lifecycle.
Oversized captured frames remain a capture-quality diagnostic and are never
used as a ground-truth signal.

Earlier broad captures can be repaired without the PCAP when the raw packet
sequence exists:

```bash
python3 repair_proxy_sequence.py
```

The repair utility removes packets that do not involve configured client IPs,
recomputes direction, reconstructs TCP flags, strips endpoint identity from the
safe sequence, and writes per-client/windowed artifacts.

`prepare_fingerprinting_dataset.py` now prefers per-client feature files when
they exist and joins `client_capture_id` to the matching federated `client_id`.
The client ID remains grouping/ground-truth metadata and is prohibited from the
predictor matrix.

Federated client output filenames now include the client ID, for example
`EXP_client_1_ground_truth.jsonl` and `EXP_client_1_resource.csv`. This avoids
file collisions when logs from several client machines are copied to one
analysis directory.

## Experiment ID collision protection

Version 0.8.2 refuses to append a restarted experiment to old output files.

The default configuration is:

```yaml
experiment:
  experiment_id: auto
  output_dir: experiments/results
  existing_output_policy: error
```

When `python3 main.py` is used interactively and output files already exist for
the same experiment ID and role, the program displays the files and requires
one of:

```text
1. use_new_experiment_id
2. archive_existing_run
3. cancel
```

Archiving moves the prior role-specific files to:

```text
<output_dir>/
└── _archive/
    └── <experiment_id>/
        └── <role>/
            └── <timestamp>/
```

The collision check is role-specific. Therefore client, server, and proxy may
and should use the same experiment ID for the same coordinated run.

For configuration-driven runs, the safe default is to stop on collision. To
explicitly archive an earlier run before starting, set:

```yaml
experiment:
  existing_output_policy: archive
```

If any participant fails during a coordinated FL experiment, treat that run as
incomplete. Restart the server, clients, and proxy with one new shared
experiment ID rather than restarting only one participant under the old run.

## Large federated model transfers

The TCP connection timeout is used only during connection establishment.
Established experiment sockets use blocking I/O so large FL model downloads
and uploads are not incorrectly terminated after 30 seconds.

This matters for large architectures such as ResNet-101, whose serialized
parameter state can be well over 100 MiB.

For synchronous federated learning, `federated.expected_clients` on the server
must equal the number of clients participating in the round. If the server is
configured for five clients, each submitted client update waits until all five
updates arrive before the round acknowledgement is returned.

When using the inline proxy, clients connect to the proxy address. The proxy's
`upstream_host`/`upstream_port` point to the real server.

## Enforced fingerprinting data isolation

The code now enforces the research threat model rather than relying only on
documentation.

```text
Proxy network observations  -> X
Client/server ground truth   -> Y
Client/server resources      -> performance analysis only
```

The proxy writes both:

```text
EXP_001_packet_sequence.csv
EXP_001_fingerprint_sequence.csv
```

The first is the raw audit sequence. It contains addresses, ports, and absolute
timestamps and is not classifier eligible.

The second is the classifier-safe sequence and excludes:

```text
timestamp_epoch
src_ip
dst_ip
src_port
dst_port
```

The handcrafted dataset builder physically separates predictors and labels:

```bash
python3 prepare_fingerprinting_dataset.py
```

No arguments are required. It discovers matching `*_features.csv` proxy files
and `*_ground_truth.jsonl` files and writes:

```text
fingerprinting_dataset/
├── fingerprinting_X_proxy.csv
├── fingerprinting_Y_ground_truth.csv
└── fingerprinting_schema.json
```

`X_proxy.csv` contains only proxy-derived network predictors plus non-predictor
grouping metadata. `Y_ground_truth.csv` contains the target labels.

The strict loader
`ai_fingerprint.fingerprinting_dataset.load_xy()` returns only the
schema-declared proxy predictor columns. It cannot silently include
experiment IDs, ground-truth labels, CPU/GPU/memory/power/energy telemetry, or
other client/server fields.

The builder also rejects `deployment=local` as a network fingerprinting sample
because purely local training/inference does not create workload exchange
visible to a network-side proxy.

See `FINGERPRINTING_DATA_POLICY.txt`.

## Inline label-blind proxy

Version 0.7 contains a dedicated proxy implementation:

```text
ai_fingerprint/proxy.py
```

The experimental topology is:

```text
Client
  |
  v
Label-blind Proxy / Gateway
  |
  v
Real Server
```

The proxy performs raw bidirectional TCP forwarding. It does not terminate TLS
or inspect the application payload:

```text
client encrypted bytes -> proxy -> real server
real server encrypted bytes -> proxy -> client
```

Run the program with no arguments:

```bash
python3 main.py
```

and select:

```text
Select experiment node role
  1. client
  2. server
  3. proxy
```

When `proxy` is selected, the program asks only for network/capture information
such as the proxy listen address, real upstream server, client-facing capture
interface, and client IP. It does NOT ask for framework, family, architecture,
variant, application, dataset, task, or device labels.

The proxy configuration is written to:

```text
proxy_config.yaml
```

and is intentionally label blind.

A normal proxy run produces:

```text
EXP_001.pcapng
EXP_001_packet_sequence.csv
EXP_001_fingerprint_sequence.csv
EXP_001_features.csv
EXP_001_manifest.json
EXP_001_proxy_forwarding.csv
EXP_001_proxy_summary.json
```

`*_proxy_forwarding.csv` records byte-stream forwarding events for diagnostics.
Its socket `recv()` chunk sizes are not network packet sizes and must not be
used as a substitute for the PCAP-derived packet sequence.

For clean fingerprinting measurements, capture only the client-facing leg:

```text
Client <-> Proxy
```

rather than both:

```text
Client <-> Proxy <-> Real Server
```

The recommended proxy capture filter is therefore:

```text
host <client_ip> and port <proxy_listen_port>
```

See `PROXY_SCHEMA.txt` and `proxy.example.yaml`.

## Main research outputs per proxy capture

A normal proxy capture now produces four synchronized files:

```text
EXP_001.pcapng
EXP_001_packet_sequence.csv
EXP_001_fingerprint_sequence.csv
EXP_001_features.csv
EXP_001_manifest.json
```

The PCAPNG file preserves the raw traffic.

The packet sequence CSV supports learned sequence models such as Transformers, Temporal CNNs, BiLSTMs, and state space models.

The feature CSV supports Random Forest, XGBoost, MLP, hierarchical classifiers, and the handcrafted branch of a hybrid model.

The manifest records the capture hash, packet count, extraction parameters, output paths, schema versions, and direction reference. It deliberately excludes AI ground truth labels.

## Packet sequence fields

The packet sequence CSV contains:

```text
experiment_id
packet_index
timestamp_epoch
relative_time_sec
frame_length
direction
src_ip
dst_ip
src_port
dst_port
transport_protocol
tcp_flags_hex
tcp_syn
tcp_ack
tcp_fin
tcp_rst
retransmission
tls_record_count
tls_record_lengths
```

Direction uses the research convention:

```text
up      client to server
down    server to client
unknown direction could not be resolved
```

Provide `--server-ip` whenever possible. When `--server-ip` is omitted during capture, `--host` is treated as the server direction reference.

## Handcrafted feature categories

The feature extractor computes more than eighty numeric features from the observable traffic.

Volume features include total packet count, upload packet count, download packet count, unknown direction packet count, total bytes, upload bytes, download bytes, and unknown direction bytes.

Directional features include upload to download packet ratio, upload to download byte ratio, upload and download packet fractions, upload and download byte fractions, direction switch count, and direction switch rate.

Throughput features include packets per second, bytes per second, upload packets per second, download packets per second, upload bytes per second, and download bytes per second.

Packet size statistics include mean, median, standard deviation, minimum, maximum, quartiles, percentiles, entropy, skewness, and kurtosis. Compact directional packet size statistics are also generated separately for upload and download traffic.

Timing features include flow duration and interarrival time mean, median, standard deviation, minimum, maximum, quartiles, percentiles, entropy, skewness, and kurtosis.

Burst features include total burst count, upload burst count, download burst count, burst frequency, and statistics for burst bytes, packet counts, durations, and intervals.

Idle features include idle gap count, total idle time, mean idle time, and maximum idle time.

Transport features include TCP packet count, UDP packet count, SYN count, ACK count, FIN count, RST count, TCP retransmission count, and connection count.

TLS features include observable TLS record count and record size statistics when tshark can decode TLS record metadata.

## Automatic capture and extraction

Run on the proxy:

```bash
python main.py capture \
    --interface wlan0 \
    --host 192.168.137.10 \
    --server-ip 192.168.137.10 \
    --port 5000 \
    --experiment-id EXP_001
```

Press Ctrl+C when the experiment completes.

The capture process is stopped gracefully, then the code automatically creates:

```text
captures/EXP_001.pcapng
captures/EXP_001_packet_sequence.csv
captures/EXP_001_features.csv
captures/EXP_001_manifest.json
```

## Extract from an existing PCAP

```bash
python main.py extract \
    --pcap captures/EXP_001.pcapng \
    --experiment-id EXP_001 \
    --server-ip 192.168.137.10
```

## Optional time window features

To generate one overall feature row plus one row per five second window:

```bash
python main.py extract \
    --pcap captures/EXP_001.pcapng \
    --experiment-id EXP_001 \
    --server-ip 192.168.137.10 \
    --window-seconds 5
```

The feature CSV then contains:

```text
row_type = overall
row_type = window
```

with `window_index`, `window_start_sec`, and `window_end_sec`.

## Burst and idle thresholds

The default burst gap is 0.05 seconds.

The default idle threshold is 0.5 seconds.

They can be changed:

```bash
python main.py extract \
    --pcap captures/EXP_001.pcapng \
    --experiment-id EXP_001 \
    --server-ip 192.168.137.10 \
    --burst-gap-sec 0.10 \
    --idle-threshold-sec 1.0
```

These values are recorded in the manifest for reproducibility.

## Capture without automatic extraction

```bash
python main.py capture \
    --interface wlan0 \
    --host 192.168.137.10 \
    --port 5000 \
    --experiment-id EXP_001 \
    --no-extract
```



## Experiment task and deployment

The experiment now separates the workload task from its deployment.

```text
Task
├── Inference
│   ├── Local
│   └── Remote
└── Training
    ├── Local
    ├── Remote
    └── Federated
```

The corresponding YAML is:

```yaml
execution:
  task: training
  deployment: federated
```

`training + remote` is centralized training. The client sends training inputs
and labels to the server, and the server performs optimization.

`training + federated` keeps raw training data at the clients. Each synchronous
round has the observable phases:

```text
Download -> Training -> Upload
```

The built in federated coordinator implements weighted FedAvg and waits for
`federated.expected_clients` updates before advancing the round.

```yaml
federated:
  rounds: 10
  local_epochs: 1
  steps_per_epoch: 10
  expected_clients: 2
  client_id: client_1
  aggregation: fedavg
```

Training ground truth includes epoch, batch, step, loss, accuracy when
applicable, learning rate, processing time, samples processed, and remote
request/response byte counts when applicable.

Federated ground truth additionally includes round, client ID, phase, model
download bytes, model upload bytes, local training metrics, and phase timing.

Training currently requires the native PyTorch or TensorFlow backend for a
variant with a native training implementation. ONNX Runtime, TensorRT, and
TFLite remain inference runtimes.

The exact contract is in `EXECUTION_SCHEMA.txt`.

## Ground truth logging

The client and server write JSON Lines ground truth logs containing:

```text
experiment_id
role
framework
runtime
family
architecture
variant
application
dataset
dataset_split
device
operating_system
task
deployment
execution_mode
precision
batch_size
input_size
event
timestamp_utc
```

Remote inference client events also include:

```text
request_id
sequence_index
request_bytes
response_bytes
round_trip_ms
output_shape
```

Server inference events include:

```text
request_id
input_bytes
output_bytes
```

The proxy output does not contain these AI labels. Joining ground truth and proxy features should be done later by `experiment_id`.



## Client and server resource telemetry

Resource monitoring is enabled by default and starts immediately before the selected workload. It stops after the workload and writes two local sidecar files:

```text
<experiment_id>_<role>_resource.csv
<experiment_id>_<role>_resource_summary.json
```

The telemetry is never transmitted through the measured client server channel.

The resource CSV contains:

```text
experiment_id
timestamp_utc
timestamp_monotonic_ns
relative_time_sec
sample_index
role
device
sample_interval_ms
telemetry_source
network_interface

bytes_sent
bytes_received
bytes_sent_total
bytes_received_total
bytes_sent_delta
bytes_received_delta

cpu_usage_percent
process_cpu_usage_percent_raw
system_cpu_usage_percent

memory_usage_mb
memory_usage_percent
system_memory_usage_percent

gpu_usage_percent
gpu_memory_used_mb
gpu_memory_total_mb

cpu_power_w
gpu_power_w
system_power_w

cpu_energy_j
gpu_energy_j
system_energy_j
```

`bytes_sent` and `bytes_received` are cumulative experiment bytes measured from the selected network interface counter baseline. The delta columns contain interval traffic.

`cpu_usage_percent` is the current experiment process CPU usage normalized to total logical CPU capacity. `process_cpu_usage_percent_raw` retains psutil semantics where 100 percent corresponds to one fully utilized logical CPU. `system_cpu_usage_percent` records whole machine utilization.

`memory_usage_mb` is the experiment process resident memory. The process and system memory percentage columns are retained separately.

Power and energy are best effort hardware measurements. On conventional Linux systems, CPU package energy is read from RAPL or AMD powercap when exposed. On NVIDIA systems, GPU utilization, memory, and power are queried through `nvidia-smi` when available. On Jetson systems, the monitor also checks common GPU devfreq and INA3221/hwmon power rails. Unsupported metrics are left empty rather than fabricated.

Energy is cumulative from the beginning of the resource monitor. Direct hardware energy counters are preferred. When only power is available, energy is integrated over time using the sampled power values.

The exact schema is documented in:

```text
RESOURCE_SCHEMA.txt
```

### Resource configuration

```yaml
resource_monitor:
  enabled: true
  interval_ms: 500
  network_interface: null
  gpu_index: 0
  power_enabled: true
```

For clean network byte measurements, set `network_interface` to the actual experiment interface such as:

```yaml
resource_monitor:
  network_interface: wlan0
```

If it is left null, psutil reports aggregate counters across all interfaces.

A 500 ms sampling interval is the default. For very lightweight workloads, 1000 ms can reduce telemetry overhead. Values below 100 ms are rejected.

### Resource summary

At experiment completion, the summary JSON reports total experiment bytes together with mean, minimum, and maximum CPU, memory, GPU, and power measurements plus final cumulative energy values.

The client and server ground truth logs also receive a `resource_summary` event containing the resource file paths, bytes sent and received, and available energy totals.

## Dataset manager

The current dataset catalog contains more than thirty configurations across image classification, object detection, image segmentation, human activity recognition, and text classification.

List them:

```bash
python main.py datasets list
```

Inspect one:

```bash
python main.py datasets info --name cifar10
```

Download one:

```bash
python main.py datasets download \
    --name cifar10 \
    --root datasets
```

Prepare all automatic datasets up to the medium size tier:

```bash
python main.py datasets download \
    --all \
    --root datasets \
    --max-tier medium
```



## Run with no arguments

The default workflow requires no command-line arguments:

```bash
python3 main.py
```

This launches the interactive experiment setup, saves the resulting
configuration to `config.yaml`, validates it, and immediately starts the
selected client or server role.

The interactive sequence asks for the experiment task and deployment first:

```text
Task
├── Inference
│   ├── Local
│   └── Remote
└── Training
    ├── Local
    ├── Remote
    └── Federated
```

It then asks for framework, runtime, family, architecture, variant,
application, dataset, device, network settings, training/inference parameters,
and resource-telemetry settings as applicable.

The explicit subcommands remain available for reproducible scripted runs, but
they are optional.

## Interactive configuration

Configure and save:

```bash
python main.py configure --output config.yaml
```

For compatibility with the workflow used during development, configure and
immediately run the selected client or server with:

```bash
python main.py --interactive
```

The menus are conditional. A selected model family determines valid architectures. An architecture determines valid applications. An application determines compatible datasets.

## Run a server

```bash
python main.py run --config server.example.yaml
```

## Run a client

```bash
python main.py run --config client.example.yaml
```

## Required proxy software

PCAP parsing requires `tshark`.

Traffic capture can use `dumpcap` or `tshark`.

Both are normally installed with Wireshark packages.

## Methodological isolation

For architecture fingerprinting, hold dataset, application, input size, batch size, transport, device, and runtime fixed while changing architecture.

For framework fingerprinting, hold architecture, application, dataset, device, batch size, and input size fixed while changing framework or runtime.

For device fingerprinting, hold framework, runtime, architecture, application, dataset, batch size, and input size fixed while changing only the device.

For the hybrid model, use the packet sequence CSV as the learned sequence branch and the feature CSV as the handcrafted feature branch.

## Project layout

```text
main.py
prepare_fingerprinting_dataset.py
config.example.yaml
server.example.yaml
client.example.yaml
ai_fingerprint/
    capture.py
    cli.py
    client.py
    proxy.py
    config.py
    dataset_catalog.py
    dataset_manager.py
    datasets.py
    fingerprinting_dataset.py
    metadata.py
    protocol.py
    registry.py
    runner.py
    server.py
    traffic/
        __init__.py
        analysis.py
    workloads/
tests/
scripts/
```


## Hierarchical model taxonomy

Version 0.6 retains the three level model identity taxonomy:

```text
family -> architecture -> variant
```

Examples:

```text
cnn -> resnet -> resnet50
rnn -> lstm -> bilstm_2layer
transformer -> vit -> vit_b16
autoencoder -> variational_autoencoder -> beta_vae
gnn -> gat -> gat_8head
diffusion -> latent_diffusion -> stable_diffusion_v1_5
gan -> stylegan -> stylegan2_ada
mlp -> feedforward_mlp -> mlp_4layer
state_space -> mamba -> mamba_small
```

The registry is data driven and is not limited to CNN, RNN, and Transformer.
CNN, RNN, Transformer, and the convolutional autoencoder variant have native
workload support where implemented by the selected framework. Broader families
remain cataloged for artifact backed inference through supported runtimes when
a compatible model artifact is supplied.

The interactive configuration now asks for:

```text
framework
runtime
family
architecture
variant
application
dataset
```

Ground truth JSONL records include `variant` as a first class field.

Legacy version 0.4 configurations such as:

```yaml
family: cnn
architecture: resnet18
```

are upgraded on load to:

```yaml
family: cnn
architecture: resnet
variant: resnet18
```

The full hierarchy contract is documented in `MODEL_HIERARCHY_SCHEMA.txt`.


## Training examples

Local training:

```bash
python main.py run --config training_local.example.yaml
```

Remote centralized training uses two terminals or machines.

Server:

```bash
python main.py run --config training_remote_server.example.yaml
```

Client:

```bash
python main.py run --config training_remote_client.example.yaml
```

Federated training uses one server plus the configured number of clients.

Server:

```bash
python main.py run --config training_federated_server.example.yaml
```

Each client uses a client configuration with a unique:

```yaml
federated:
  client_id: client_1
```

Then run:

```bash
python main.py run --config training_federated_client.example.yaml
```

Copy the client configuration for additional clients and change only the
client ID, device label, dataset partition, and server address as appropriate.
