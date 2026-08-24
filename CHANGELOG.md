# Changelog


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
