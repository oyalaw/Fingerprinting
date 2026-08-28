# 0.9.3

- Removed the interactive proxy storage-locator prompt. The proxy no longer receives `family/architecture/variant/application/dataset/framework/expN` during capture.
- Added a server-side out-of-band experiment coordinator on `10.42.0.195:8081` by default. It exposes only a neutral per-execution `run_id`; it does not expose AI labels.
- Added automatic proxy run discovery. The proxy contacts the coordinator, obtains the neutral `run_id`, and writes to `experiments/staging/<run_id>/proxy/`.
- Added neutral `run_id` metadata to server manifests/status files and to the federated policy response so clients can record the same execution identity after connecting.
- Kept the server/client scientific hierarchy unchanged at `family/architecture/variant/application/dataset/framework/expN/`. The neutral run ID is a cross-node correlation key, not a classifier feature.
- Added regression coverage for neutral-ID generation, coordinator discovery, label-free proxy staging, and coordination defaults.

Validation: 114 automated tests pass; Python compilation and package integrity checks pass.

# 0.9.2

- Removed the normal interactive prompt for comma-separated participating client IPs on the proxy.
- Added automatic participating-client discovery from actual accepted proxy connections; discovered IPs are grouping/capture-isolation metadata only and never classifier predictors.
- Tightened automatic archival and live BPF capture to the proxy client-facing endpoint and listen port while explicitly excluding the configured upstream FL server, e.g. `host 10.42.0.1 and port 8080 and not host 10.42.0.195`.
- Added dynamic live-monitor client registration so real-time 0.5/1/2/5-second features can begin without a pre-entered client allow-list.
- Added neutral stable aliases (`trace_001`, `trace_002`, ...) as clients are discovered and reused the same mapping for end-of-run per-client artifacts.
- Automatic capture-interface selection now uses the proxy listen IP first; manual interface selection remains only as an ambiguity fallback.
- Retained legacy/manual `capture.client_ips` support through `capture.client_discovery_mode: manual` for restricted/diagnostic captures.
- Extended capture manifests and proxy summaries with discovery method, discovered clients, aliases, excluded upstream server, and the actual BPF filter used.

Validation: 110 automated tests pass, including new automatic-discovery, upstream-exclusion, alias-stability, and live-registration regression tests.

# 0.9.1

- Made the FL server the single authority for six controlled training parameters: input size, batch size, learning rate, global rounds, local epochs per round, and local steps per epoch.
- Removed those six prompts from federated clients. The client now requests a server training policy immediately after connecting and applies it before creating its dataset generator or workload.
- Added `fl_policy_get` handshake plus a run-salted policy digest. `fl_get` and `fl_update` are rejected if the client has not applied the current server policy.
- Changed client global-round control from a local `range(rounds)` loop to server-driven execution until the server explicitly reports `done=True`.
- Added `server_training_policy.json` and `config_effective.yaml` to client role output so the actual server-issued values are preserved.
- Added policy identifiers and server-authoritative training controls to per-round client/server performance logs.
- Prevented pre-handshake client ground truth from falsely reporting placeholder input-size/batch-size values; those fields are marked pending until policy synchronization completes.
- Added regression tests for policy integrity, server policy delivery, deferred client generator creation, interactive prompt ownership, and server/client policy application.

Validation: 104 automated tests pass. A one-round local end-to-end synchronous FL smoke test also verified that deliberately wrong client placeholder values are overwritten by the server policy before model construction/training.

# 0.9.0

- Added per-round federated client `round_metrics.csv` with task-aware training loss, accuracy/precision/recall/F1 or reconstruction MSE/MAE/KL metrics.
- Added optional held-out round probe before/after local client training and improvement deltas.
- Added client/global model L2 norms and true local update norm `||W_i^t-W^t||_2`.
- Added server `client_update_metrics.csv` with one row per client per round.
- Added server `round_metrics.csv` after every FedAvg aggregation, including global loss/accuracy/precision/recall/F1, reconstruction metrics, convergence deltas, aggregation/evaluation time, participation, model/update norms and byte totals.
- Server evaluation resets to the same evaluation samples each round for comparable convergence trajectories.
- Added interactive defaults for enabling performance logging and choosing server evaluation batches/split.
- Explicitly forbids training-performance metrics and model/update norms from proxy fingerprinting predictors.
- Corrected autoencoder real-dataset targets: reconstruction/anomaly/image-denoising workloads now use the input image as the target rather than a dataset class label.
- Added v0.9.0 regression tests.

# Changelog

## 0.8.9

- Added hierarchical experiment storage: `family/architecture/variant/application/dataset/framework/expN/<role>/`.
- Added branch-local automatic experiment numbering using `max(existing expN)+1`; deleted/failed numbers are never silently reused.
- Server/client interactive workflows print a coordinated storage locator; the label-blind proxy accepts that locator only as operator filesystem metadata so its files land under the same experiment hierarchy without entering predictor features.
- Added role `config.yaml`, `role_manifest.json`, `experiment_status.json`, a common `experiment_manifest.json`, and an `analysis/` directory under each run.
- Added route-based automatic network-interface detection with local-IP and interactive numbered fallbacks; manual typing of names such as `wlx0013eff408bf` is no longer required during normal runs.
- Added testbed defaults: client -> proxy `10.42.0.1:8080`, proxy -> FL server `10.42.0.195:8080`, server bind `10.42.0.195:8080`. Pressing Enter accepts them.
- Changed the normal proxy workflow to one-keystroke multi-scale collection at 0.5/1/2/5 seconds. Single-scale capture remains available only as an explicit ablation choice.
- Added a federated model-contract fingerprint over hierarchy, model input/class settings, precision, tensor count, shapes, and dtypes. Server rejects mismatched clients before sending global weights, preventing 4-layer/6-layer parameter-count failures from reaching `set_parameters()`.

Validation: 89 automated tests pass.

## 0.8.8

- Replaced free-text OS entry in the interactive client/server workflow with a numbered operating-system menu.
- Added canonical operating-system labels and device-aware OS choices in the registry.
- Kept `custom` as an explicit fallback for unsupported research platforms.
- Preserved hardware/OS separation so labels such as `ubuntu` cannot accidentally be used as the device identity.

## 0.8.7

Corrected hierarchical model availability and native execution. Transformer
now includes Tiny Transformer 2/4/6-layer variants, PyTorch BERT/DistilBERT,
and PyTorch ViT. Autoencoder now exposes Dense, Convolutional, and Variational
architectures with multiple native variants. MLP 2/4/8-layer variants are now
native in PyTorch and TensorFlow.

Added an explicit model catalog command (`python main.py models`) so
artifact-only families/architectures are visible rather than silently hidden
from native-training menus.

Completed legacy concrete-model migration for all registered variants, added
Dell desktop/laptop hardware labels, and reject OS names such as `ubuntu` when
used incorrectly as `device.label`.

Added stage-specific top-10 multiclass Fisher-score feature selection to the
hierarchical Family -> Architecture -> Variant classifier. Grouped evaluation
now reports accuracy, balanced accuracy, macro precision, macro recall, macro
F1, and log loss. Training writes `hierarchical_metrics.csv`.

Added a fail-closed multiscale guard so real-time architecture inference does
not silently fall back to a persisted 5-second-only configuration.

Validation: 77 automated tests pass. New PyTorch training smoke tests cover
Tiny Transformer 4-layer, convolutional Autoencoder 6-layer, and MLP 4-layer;
additional manual smoke checks exercised all new built-in Tiny Transformer,
Autoencoder/VAE, MLP variants and ViT-B/16 and ViT-B/32.

## 0.8.6

Added progressive real-time and complete-trace hierarchical fingerprinting.

- Added multi-scale 0.5/1/2/5-second proxy feature extraction.
- Reworked window extraction to linear-time packet binning for million-packet
  captures.
- Added a live client-facing `tshark` metadata monitor that uses only
  proxy-observable packet information and does not terminate TLS.
- Added stable-confidence real-time Family -> Architecture -> Variant
  predictions.
- Added complete-trace final predictions when the archival PCAP is extracted.
- Added `full` and `size_normalized` model configurations.
- Added Random-Forest hierarchy bundles with probability outputs and top
  feature importance metadata.
- Added group-aware evaluation by experiment ID and explicit
  `insufficient_independent_runs` status.
- Added no-argument `train_fingerprinting_models.py`.
- Added backward compatibility for legacy 5-second feature files.
- Added `window_size_sec` as non-predictor metadata.
- Added strict federated client/server experiment-ID matching.
- Interactive runs now require an explicit coordinated experiment ID.
- Proxy/client/server default output paths are aligned to
  `experiments/results`.


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
