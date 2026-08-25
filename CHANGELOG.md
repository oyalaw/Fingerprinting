# Changelog

## 0.8.5

Integrated corrections derived from the two-client federated experiment.

- Split client `Upload` from synchronous post-upload waiting. The network-send
  interval remains `Upload`; round-release waiting is now `Idle` with
  `reason=synchronous_round_wait`.
- Added `federated_upload_transaction` records with transfer, synchronization
  wait, and total transaction timing.
- Server Upload ground truth now marks `boundary=receive_complete` before model
  deserialization and logs coordinator wait separately.
- Added UTC epoch and monotonic timestamps to every ground-truth event.
- Federated clients now log `network_registration` with actual client ID and
  local source IP.
- Proxy fallback aliases are neutral `trace_###` names rather than names that
  can be mistaken for federated client IDs.
- Dataset preparation automatically resolves proxy trace IDs to actual FL
  client IDs by joining proxy-manifest client IPs with client-side
  network-registration ground truth.
- Added global alignment metadata to per-client features while keeping those
  fields outside the predictor matrix.
- Resource telemetry now records actual interval, sampling duration, and
  sampling overrun.
- `nvidia-smi` is probed once and disabled when no usable GPU is present,
  preventing repeated failed GPU queries from stretching CPU-only sampling.
- Interactive network experiments now offer TLS first.
- Added automatic OpenSSL-based self-signed research certificates for TLS
  servers when no certificate/key is supplied.
- Enforced a configurable TLS minimum version, defaulting to TLS 1.2.



## 0.8.4

Added active packet-size fidelity protection for Linux capture interfaces.

- Added `CaptureOffloadManager` for GRO/GSO/TSO/LRO lifecycle management.
- Reads and records the original `ethtool -k` state before capture.
- Disables only enabled, mutable capture-affecting offloads.
- Verifies the disabled state before packet capture begins.
- Required captures fail closed when the state cannot be verified.
- Stops PCAP capture before restoring interface settings.
- Restores exactly the settings changed by the experiment and verifies the
  final state.
- The Python application remains unprivileged; only individual `ethtool -K`
  commands may use `sudo -n`.
- Added `scripts/install_offload_sudoers.sh`, which installs a narrow sudoers
  rule limited to one user, one interface, GRO/GSO/TSO/LRO, and on/off.
- Added `OFFLOAD_PRIVILEGES.md`.
- Each proxy run writes `<experiment_id>_proxy_offload_state.json` and embeds
  the offload lifecycle in provenance metadata.
- Standalone `capture` mode uses the same offload lifecycle by default.



## 0.8.3

Hardened inline-proxy capture for multi-client fingerprinting experiments.

- Capture now requires one or more participating client IPs by default and
  builds a client-only BPF filter, excluding the proxy-to-upstream duplicate
  traffic leg even when both legs use the same interface and port.
- Added per-client classifier-safe packet sequences and handcrafted feature
  files. Combined multi-client traces are retained for audit but marked
  classifier-ineligible.
- Feature extraction now uses an overall row plus 5-second windows by default
  in the proxy workflow.
- PCAP capture defaults to a 256-byte snapshot length, retaining original frame
  lengths while avoiding storage of full encrypted payloads; set 0 for full
  frames.
- Per-chunk proxy forwarding CSV logging is disabled by default; aggregate
  forwarding counters remain in the proxy summary, reducing unnecessary
  diagnostic output size.
- TCP SYN/ACK/FIN/RST fields are derived from the authoritative `tcp.flags`
  bitmask instead of relying only on tshark subfields.
- Added MTU and GRO/GSO/TSO/LRO capture preflight metadata plus oversized-frame
  diagnostics for possible host offload/coalescing artifacts. Interface
  settings are never modified automatically.
- Added `repair_proxy_sequence.py`, which can clean an earlier broad capture
  from its raw packet-sequence CSV without requiring the original PCAP.
- Added `client_capture_id` as non-predictor metadata and multi-client
  ground-truth joining against the exact federated `client_id`.
- Client federated ground-truth records now include `client_id` in the base
  metadata so per-client joins are unambiguous.
- Federated client ground-truth/resource filenames now include the client ID,
  avoiding filename collisions when multiple client logs are consolidated.
- `prepare_fingerprinting_dataset.py` prefers per-client feature files when
  available and excludes the mixed combined feature file from classifier
  dataset construction.



## 0.8.2

Added fail-closed experiment-output protection.

Before a client, server, or proxy run starts, the code checks whether the same
experiment ID already has output files for that role. The default policy is
`error`, which refuses to append to or mix with the previous run.

The no-argument interactive workflow now offers three choices when a collision
is detected:

1. use a new experiment ID;
2. archive the existing run and continue;
3. cancel.

Archived files are moved under:

`<output_dir>/_archive/<experiment_id>/<role>/<timestamp>/`

The same experiment ID can still be shared across client, server, and proxy;
collision detection is role-specific.

Scripted/config-driven runs can explicitly set:

`experiment.existing_output_policy: archive`

to archive prior role-specific outputs automatically. The default remains
`error`.



## 0.8.1

Fixed federated transfer timeouts for large models.

The 30-second socket timeout is now used only while establishing the TCP
connection. Once connected, client/server data transfer uses blocking socket
I/O with TCP keepalive, so large global models and client updates are not
aborted after 30 seconds.

Added federated upload-size/throughput diagnostics on the client and explicit
`expected_clients` information at server startup. In synchronous FL the server
does not acknowledge a round update until every configured client has
submitted.



## 0.8.0

Enforced proxy-only attacker predictors.

Added `fingerprinting_dataset.py` with strict proxy-feature validation,
physically separated X/Y dataset construction, resource-field rejection,
role-aware ground-truth resolution, and a strict downstream `load_xy()`.

Added classifier-safe packet sequences that exclude absolute timestamps,
IP addresses, and ports while retaining the raw sequence for audit.

Added the no-argument `prepare_fingerprinting_dataset.py` utility and
`FINGERPRINTING_DATA_POLICY.txt`.

Local deployment is rejected as a network-side fingerprinting sample because
pure local execution has no workload exchange observable at the proxy.



## 0.7.0

Added `ai_fingerprint/proxy.py`, a dedicated label-blind inline TCP/TLS
forwarder.

The no-argument workflow now starts by selecting client, server, or proxy.

The proxy automatically captures the client-facing traffic when enabled,
forwards TLS without termination/decryption, extracts packet sequences and
handcrafted features after shutdown, and writes forwarding diagnostics and a
proxy summary.

Added `proxy.example.yaml` and `PROXY_SCHEMA.txt`.



## 0.6.1

`python3 main.py` now starts the interactive configuration and immediately runs
the selected experiment. No command-line arguments are required.

The older `python3 main.py --interactive` entry point remains supported.


## 0.6.0

Added explicit experiment task and deployment taxonomy.

```text
Inference: local, remote
Training: local, remote, federated
```

Added native local training and centralized remote training for supported
PyTorch and TensorFlow workloads.

Added synchronous federated training with weighted FedAvg, including the
Download, Training, and Upload phase labels.

Added training batch serialization and target-aware dataset sampling.

Added parameter export/import for native PyTorch and TensorFlow workloads.

Added a native convolutional autoencoder variant for reconstruction and
anomaly-detection experiments.

Ground-truth logs now include `task`, `deployment`, `family`,
`architecture`, and `variant`.

Retained client/server resource telemetry including bytes sent/received, CPU,
GPU, memory, power, and energy.

Added `python main.py --interactive` as a configure-and-run entry point.

Validation:
29 automated tests pass. A PyTorch local-training smoke test, centralized
remote-training smoke test, and one-round synchronous FedAvg smoke test were
also exercised successfully in the build environment.
