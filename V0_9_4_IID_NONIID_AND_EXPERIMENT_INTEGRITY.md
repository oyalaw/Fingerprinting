# v0.9.4 — IID/non-IID partitioning and experiment-integrity fixes

This release addresses the issues observed in the three-client Autoencoder smoke test and adds explicit server-controlled IID/non-IID federated data experiments.

## 1. Server-controlled IID/non-IID selection

The FL server now asks:

```text
Select federated data distribution
  1. iid
  2. non_iid
```

For `non_iid`, it then asks for the Dirichlet concentration, default `alpha=0.5`. Clients do not choose the partition regime independently.

The partition seed is automatic. With the default execution seed 42, `exp1` uses 42, `exp2` uses 43, `exp3` uses 44, and so on. Thus repeated runs receive independently shuffled assignments while the same expN can be reproduced across model branches.

### IID

The training split is shuffled with the partition seed and divided into disjoint, approximately equal client shards. Every training sample belongs to exactly one client.

### non-IID

The implementation uses class-wise Dirichlet label skew. For every class k, the allocation proportions across clients are sampled from a symmetric Dirichlet distribution with concentration alpha. Smaller alpha creates stronger client heterogeneity. Assignments remain disjoint.

For Autoencoder workloads on labeled image datasets such as Fashion-MNIST, the class label is used only to create the client input distribution. The learning target remains reconstruction `x -> x`; class labels are never used as reconstruction targets.

## 2. Round-0 all-client readiness barrier

The server does not release the first global model until all configured clients have reached the first `fl_get` after policy synchronization and model construction. The request blocks at the server rather than producing polling traffic.

Therefore operator startup delay is outside Round 0 and does not inflate Download/Upload/Idle timing features.

## 3. Connection-granular proxy isolation

Automatic proxy discovery records accepted TCP sessions by exact `(client IP, client source port)`, not by IP alone. A retry/stale connection from the same host therefore receives a separate neutral trace.

Example:

```text
10.42.0.145:57000 -> trace_001   # retry/stale
10.42.0.145:51161 -> trace_004   # confirmed FL client connection
```

Client ground truth now emits a post-handshake `network_registration_confirmed` event carrying the neutral `run_id`, local IP, and source port. Dataset preparation matches proxy traces to clients by exact run ID + IP + source port. If a run has confirmed mappings, unmatched connection traces are excluded from classifier input automatically.

IP addresses and ports remain grouping/audit metadata only and never enter the fingerprint predictor matrix.

## 4. Neutral run-ID join fixed

The proxy uses the neutral `run_id` as its experiment identifier, whereas clients and servers retain human-readable `expN` in their scientific hierarchy. Dataset preparation now uses the unique neutral run ID recorded in each ground-truth file as the join key, including for records written before the run ID was known.

This allows automatic joining of staged proxy traces to the correct client/server ground truth without exposing AI labels to the proxy.

## 5. Partition-specific logging

Each federated client writes:

```text
data_partition.json
round_metrics.csv
round_metrics_iid.csv
```

or, for example:

```text
data_partition.json
round_metrics.csv
round_metrics_non_iid_alpha_0p5.csv
```

The server similarly writes partition-specific copies of `round_metrics.csv` and `client_update_metrics.csv`, plus `data_partition_policy.json`.

The canonical files remain for backward compatibility. Do not concatenate the canonical and partition-specific copies as though they were independent observations; they contain the same round records.

Partition metadata includes type, alpha, seed, client slot, shard size, assignment ID/digest, and class histogram where labels are available. These values are ground-truth/experimental context and are explicitly forbidden from proxy predictor features.

## 6. Completion and manifest consistency

- Server metadata is re-materialized immediately after the neutral run ID is generated, removing `run_id: null` from server manifests.
- Clients re-materialize their metadata after receiving the server run ID and partition policy.
- After the final round acknowledgement has been sent to every expected client, the server exits normally so the runner writes `experiment_status.json` as `COMPLETE`.

## Recommended use

For the main controlled baseline, select `iid`. For the primary heterogeneity experiment, select `non_iid` with `alpha=0.5`. A later severe-skew stress test can use `alpha=0.1`.

Report IID and non-IID fingerprinting performance separately first, then evaluate cross-regime generalization (train IID/test non-IID and vice versa).
