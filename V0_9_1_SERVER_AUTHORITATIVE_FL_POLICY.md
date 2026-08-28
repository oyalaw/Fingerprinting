# v0.9.1 — Server-authoritative federated training policy

## Why this change was made

In previous versions, both the FL server and every client independently entered
several training controls. That is unnecessary for the controlled experiments
and can produce silent configuration heterogeneity. v0.9.1 makes the server the
single authority for six values:

1. image/input size;
2. batch size;
3. learning rate;
4. number of global federated rounds;
5. local epochs per round; and
6. local steps per epoch.

## Interactive behavior

The server still asks:

```text
Image input size [32]:
Batch size [1]:
Learning rate [0.001]:
Federated rounds [10]:
Local epochs per round [1]:
Local steps per epoch [10]:
```

A federated client no longer asks those questions. It only collects genuinely
client-specific controls such as client ID, device/OS, resource telemetry and
network settings. It prints:

```text
Federated training policy will be received from the server: input size,
batch size, learning rate, global rounds, local epochs, and local steps.
```

## Protocol flow

After the TCP/TLS connection is established, the client first sends
`fl_policy_get`. The server returns a versioned, run-salted training policy.
The client validates the policy, applies it to its effective configuration, and
only then creates its `InputGenerator` and trainable workload.

The client then builds the normal model contract and starts FL. Every `fl_get`
and `fl_update` includes the training-policy digest. The server rejects a
request whose policy digest does not match the current server policy.

Global-round ownership is also server-side. The client no longer loops over a
locally configured round count; it requests global models until the server
explicitly returns `done=True`.

## Files written on each federated client

In addition to the existing role artifacts, the client now writes:

- `server_training_policy.json` — exact policy received from the server;
- `config_effective.yaml` — client configuration after the server policy has
  been applied.

The initial `config.yaml` remains useful for documenting locally selected
identity/device/network settings, while `config_effective.yaml` is the
configuration actually executed for training.

## Training-performance logs

Client `round_metrics.csv` now also records:

- `input_size`
- `batch_size`
- `global_rounds`
- `local_epochs`
- `steps_per_epoch`
- `learning_rate`
- `training_policy_id`
- `policy_source=server`

Server `round_metrics.csv` records the same policy context alongside global
learning metrics.

These are ground-truth/system-characterization fields and remain forbidden as
proxy-side fingerprinting predictors.

## Validation

The v0.9.1 package passes 104 automated tests. A one-round end-to-end local FL
smoke test was also run with deliberately incorrect client placeholders
(input size 224, batch size 9, learning rate 0.123, 99 rounds, 7 local epochs,
8 local steps). Before constructing the client workload, the server policy
correctly replaced them with input size 32, batch size 1, learning rate 0.001,
1 round, 1 local epoch and 1 local step.
