# Capture-interface offload privileges

AI Fingerprinting v0.8.4 manages GRO, GSO, TSO, and LRO only on the configured
capture interface and only for the duration of an experiment.

The Python application itself should **not** be run as root.

## Why privilege is needed

Changing NIC offload settings with:

```bash
ethtool -K <interface> gro off
```

normally requires `CAP_NET_ADMIN`. The experiment first tries `ethtool`
directly. If that is denied and `allow_sudo_noninteractive: true`, it retries
only that exact `ethtool -K` operation with `sudo -n`.

`sudo -n` never opens a password prompt. If no permission exists, capture
fails closed when `required: true`.

## Recommended narrow permission

Install a sudoers rule restricted to:

- one local user;
- one capture interface;
- `ethtool -K`;
- only `gro`, `gso`, `tso`, and `lro`;
- only `on` and `off`.

From the project directory:

```bash
sudo apt install ethtool
sudo bash scripts/install_offload_sudoers.sh wlx0013eff408bf
```

Or specify a username explicitly:

```bash
sudo bash scripts/install_offload_sudoers.sh wlx0013eff408bf tiger
```

The installer validates the rule with `visudo` before installing it.

The rule does **not** permit arbitrary root commands and does not make the
whole Python process privileged.

## Experiment lifecycle

With the default proxy configuration:

```yaml
capture:
  offload_management:
    enabled: true
    required: true
    allow_sudo_noninteractive: true
    restore_on_exit: true
    features: [gro, gso, tso, lro]
```

the proxy performs:

```text
read original state
        ↓
disable enabled mutable GRO/GSO/TSO/LRO
        ↓
read state again and verify
        ↓
start packet capture
        ↓
stop packet capture
        ↓
restore exactly the settings changed by the experiment
        ↓
verify restoration
```

Settings that were already `off` remain `off`.

If a driver reports an enabled feature as `[fixed]`, or the disabled state
cannot be verified, a required capture is aborted rather than collecting
packet-size fingerprints under an uncertain kernel-offload state.

## Provenance

Each proxy run writes:

```text
<experiment_id>_proxy_offload_state.json
```

This records:

- pre-experiment state;
- each disable attempt;
- verified capture-time state;
- each restoration attempt;
- verified post-experiment state.

The same report is included in the proxy summary/manifest metadata and is not
used as a classifier predictor.
