# v0.9.5 — Anomaly metrics and full-scale rounds

## What changed

1. The interactive federated server now uses **100 rounds as the minimum**. Press Enter
   at `Federated rounds [100]` to use the full-scale default. Values below 100 are
   rejected by the interactive workflow. Clients do not choose the round count.
2. Autoencoder `anomaly_detection` is no longer evaluated as reconstruction-only.
3. The server asks for anomaly class labels (default: the last dataset class) and a
   normal calibration percentile (default: 95). The complete protocol is distributed
   to every client through the server-authoritative FL policy.
4. Configured anomaly classes are excluded from training. Remaining normal examples
   are partitioned IID or Dirichlet non-IID, preserving the existing disjoint-client
   data design. Reconstruction targets remain `x -> x`.
5. Each round recalibrates a reconstruction-MSE threshold using a fixed held-out normal
   calibration set and evaluates a fixed held-out normal+anomaly set.
6. Client round logs now populate `train_accuracy`, `train_precision`, `train_recall`
   and `train_f1` for anomaly detection. Server logs populate the corresponding
   `global_*` fields. `train_loss`/`global_loss` remain reconstruction/evaluation loss.
7. Added AUROC, AUPRC, threshold, TP/FP/TN/FN, sample counts and error means.
8. Added concise `anomaly_detection_metrics.csv` plus IID/non-IID-specific copies.
9. Added anomaly labels, thresholds and performance metrics to the forbidden proxy
   predictor policy.

## Default anomaly prompts on the server

```text
Anomaly class labels (comma-separated) [9]:
Normal-score threshold percentile [95]:
...
Federated rounds [100]:
...
Normal calibration batches per round [10]:
Anomaly evaluation batches per round [10]:
Anomaly evaluation batch size [32]:
```

For Fashion-MNIST with the default `[9]`, class 9 is held out as the anomaly class and
classes 0-8 are normal training classes. You may enter multiple anomaly labels such as
`8,9`.

## Output

Client and server both retain the normal `round_metrics.csv` and partition-specific
round files. Anomaly runs also create:

```text
anomaly_metrics.csv
anomaly_metrics_iid.csv
# or anomaly_metrics_non_iid_alpha_0p5.csv

anomaly_detection_metrics.csv
anomaly_detection_metrics_iid.csv
# or anomaly_detection_metrics_non_iid_alpha_0p5.csv
```

The concise anomaly file contains direct columns named `loss`, `accuracy`, `precision`,
`recall`, and `f1_score`, plus AUROC/AUPRC and reconstruction metrics.
