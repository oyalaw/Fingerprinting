# v0.9.3 — Neutral proxy run coordination

## Why this change exists

The scientific results hierarchy contains ground-truth labels:

`family/architecture/variant/application/dataset/framework/expN/`

The proxy should not need those labels merely to decide where to save a PCAP. v0.9.3 therefore removes the interactive storage-locator prompt.

## New workflow

1. The server allocates the normal hierarchy/`expN` and creates a neutral run ID such as `run_20260828t184455z_a1b2c3d4`.
2. The server starts a small out-of-band coordination endpoint on `10.42.0.195:8081` by default.
3. The proxy already knows the FL server IP. At startup it queries the coordination endpoint and receives only the neutral `run_id`.
4. The proxy stores its capture under `experiments/staging/<run_id>/proxy/`.
5. Server and client manifests record the same `run_id`, allowing later result collection/joining without exposing family, architecture, variant, application, dataset, or framework to the proxy.

The coordination endpoint is deliberately separate from the fingerprinted FL path and returns no AI labels.

## Normal proxy startup

The proxy now prints approximately:

```text
Experiment storage discovery: automatic (neutral run ID only).
Active neutral run ID: run_20260828t184455z_a1b2c3d4
Proxy staging directory: experiments/staging/run_20260828t184455z_a1b2c3d4/proxy
No AI family, architecture, variant, application, dataset, or framework label was supplied to the proxy.
```

There is no storage-locator prompt.

## Start order

Start the server before the proxy. If the server coordinator is not yet available, the proxy waits and retries automatically. Ctrl-C remains the escape path.

## Ports

- FL server data path: `10.42.0.195:8080`
- Experiment coordination: `10.42.0.195:8081`
- Proxy client-facing path: `10.42.0.1:8080`

Port 8081 carries only the neutral run coordination exchange and is not part of the AI fingerprint feature set.
