# Federated data-partition schema

Federated partitioning is server-authoritative and applies only to client-side training data. Server evaluation retains the full configured evaluation split.

```yaml
data:
  partition:
    type: iid            # iid | non_iid
    alpha: 0.5           # used only by non_iid Dirichlet label skew
    seed: 42             # automatic from base execution seed + expN - 1
    client_count: 3
    client_index: 0      # assigned by server to each client
    client_id: client_1
    disjoint: true
    source: server
```

IID uses a deterministic shuffled disjoint split. non-IID uses class-wise Dirichlet allocation while preserving disjointness. For reconstruction workloads, labels may shape the input partition but the target remains the input itself.

Client `data_partition.json` records the actual shard summary and index digest. Partition fields are research context only and are prohibited from classifier X.


## Anomaly detection

For `application: anomaly_detection`, configured anomaly class labels are held out of
training before IID/non-IID partitioning. The remaining normal classes are partitioned
disjointly across clients, so Dirichlet non-IID remains meaningful. Evaluation uses
the full held-out evaluation split and the anomaly labels only as ground truth.

Example:

```yaml
anomaly_detection:
  anomaly_labels: [9]
  threshold_percentile: 95.0
  calibration_batches: 10
  evaluation_batches: 10
  evaluation_batch_size: 32
```

`data_partition.json` records the anomaly labels and that they were excluded from
training. These fields are research metadata and forbidden as classifier predictors.
