# v0.9.2 — Automatic proxy participating-client discovery

## Purpose

The proxy no longer asks the operator to type a comma-separated list of participating client IP addresses for ordinary experiments. The proxy already observes accepted client TCP connections, so those peer addresses are the authoritative network/session source for capture grouping.

## Default topology

```text
Clients -> 10.42.0.1:8080 (Proxy) -> 10.42.0.195:8080 (FL Server)
```

The automatic archival/live BPF is scoped to the proxy client-facing endpoint and port and explicitly excludes the upstream server:

```text
host 10.42.0.1 and port 8080 and not host 10.42.0.195
```

This prevents the proxy-to-server duplicate leg from entering the capture without requiring client IPs before the clients connect.

## Discovery and aliases

As each client connects, the proxy records its peer IP as capture-isolation metadata and assigns a neutral stable alias in discovery order:

```text
[proxy] discovered client 10.42.0.47 -> trace_001 [accepted_connection]
[proxy] discovered client 10.42.0.210 -> trace_002 [accepted_connection]
```

The real-time metadata reader may observe SYN/handshake packets before `accept()` returns. It buffers a small number of such packets by peer and releases them into the live trace only after the proxy confirms an actual accepted connection. This avoids treating unrelated SYN scans as participating clients while preserving the start of the real connection.

## Label-blindness

Client IPs and neutral aliases are used only to isolate/group observable network traffic. They are removed from classifier-safe packet sequences and are not fingerprint predictors. Family, architecture, variant, framework, dataset, application, device, and training metrics remain unavailable to the proxy predictor pipeline.

## Offline extraction

At capture stop, the discovered client list and alias mapping are passed to the normal post-filter/extraction pipeline. Per-client safe sequences/features therefore remain isolated even though the client IPs were not known when the PCAP process started.

## Manual override

Scripted/diagnostic configurations may retain the legacy allow-list behavior:

```yaml
capture:
  client_discovery_mode: manual
  client_ips: [10.42.0.47, 10.42.0.210]
```

In manual mode, strict isolation still requires at least one configured client IP. The interactive workflow defaults to automatic discovery and does not ask for the list.

## Manifest evidence

The proxy summary and capture manifest now record the discovery mode, discovered client IPs, neutral aliases, upstream server excluded from capture, and the BPF filter used. These are audit metadata, not classifier inputs.

## Validation

The v0.9.2 package passes 110 automated tests. Regression coverage includes automatic mode with no pre-entered client IPs, manual-mode validation, upstream-server BPF exclusion, stable neutral aliases, live client registration, and buffering/flushing of pre-accept handshake packets.
