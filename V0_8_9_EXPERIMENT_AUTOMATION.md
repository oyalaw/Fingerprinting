# v0.8.9 Experiment Automation

## Storage hierarchy

Every client/server experiment uses:

`experiments/results/<family>/<architecture>/<variant>/<application>/<dataset>/<framework>/expN/<role>/`

The next default experiment number is `max(existing expN)+1` within the selected branch. Failed experiment directories remain part of the numbering history.

The server prints an **Experiment storage locator** such as:

`autoencoder/convolutional_autoencoder/convolutional_autoencoder_6layer/anomaly_detection/fashion_mnist/pytorch/exp4`

Paste that one locator into the proxy. It is filesystem organization metadata only and is never inserted into classifier-safe traffic features.

## Network defaults

- Proxy: `10.42.0.1:8080`
- FL server: `10.42.0.195:8080`
- Clients default to the proxy, not directly to the server.

Press Enter to accept each default.

## Automatic interfaces

Clients select the resource-telemetry interface from the route to `10.42.0.1`. The server maps its bind IP (`10.42.0.195`) to the local interface. The proxy first looks for route consensus to all configured client IPs, then maps its listen IP (`10.42.0.1`), and only shows a numbered manual menu if automatic detection is inconclusive.

## Window scales

Normal proxy runs ask:

`Use default multi-scale fingerprinting windows (0.5 s, 1 s, 2 s, 5 s) [Y/n]:`

Press Enter to collect all four scales. Answer `n` only for a deliberate single-scale or custom ablation.

## Federated model contract

Before the server sends global parameters, client and server compare a run-salted SHA-256 contract covering framework/runtime, family/architecture/variant/application, precision, input/class settings, tensor count, tensor shapes, and dtypes. A mismatch aborts before model transfer with a `FEDERATED MODEL CONTRACT MISMATCH` message.
