# v0.9.0 — Per-round federated learning performance logging

v0.9.0 adds task-aware learning-performance logs to both federated clients and the server. These logs are ground-truth/system-characterization artifacts only; they are explicitly excluded from proxy-side fingerprinting predictors.

## Client output

Each federated client writes `round_metrics.csv` in its role directory. One row is written per completed FL round. Classification workloads record local training loss, accuracy, macro precision, macro recall and macro F1 over all local training examples in the round. Reconstruction/autoencoder workloads record total loss, reconstruction loss, MSE, MAE and, for VAE variants, KL loss and beta.

When `performance_logging.client_round_probe=true`, the client also holds out one batch from local training for that round, evaluates it immediately after receiving the global model and again after local training, and records before/after loss, accuracy/F1 when applicable, MSE for reconstruction tasks, and improvement deltas. The probe batch is not optimized on during that round.

The client row also includes download/training/upload/synchronous-wait timing, serialized model size, global/local model L2 norms and the true local update norm `||W_i^t - W^t||_2`.

## Server output

The server writes:

- `round_metrics.csv`: one row after each FedAvg aggregation.
- `client_update_metrics.csv`: one row per received client update per round.

The global round file records task-aware evaluation metrics, aggregation and evaluation time, client participation, global model/update norms, mean/std client update norm, model size, per-round traffic byte totals and convergence deltas.

By default, the server evaluates the same first 10 batches of the `test` split after each aggregation. The evaluation generator is reset each round so model-performance trends are not confounded by a changing evaluation sample. This server evaluation happens before synchronous clients are released, and `evaluation_time_ms` is logged explicitly.

## Interactive defaults

For federated training the CLI now asks:

```text
Log per-round training performance [Y/n]:
```

Client:

```text
Evaluate one held-out probe batch before/after each local round [Y/n]:
```

Server:

```text
Global evaluation batches after each federated round [10]:
Global evaluation dataset split [test]:
```

Pressing Enter accepts the defaults.

## Data-isolation rule

`round_metrics.csv`, `client_update_metrics.csv`, losses, accuracy, precision, recall, F1, reconstruction metrics, model/update norms, aggregation time and evaluation time are forbidden as proxy fingerprinting predictors. They may be used only for ground truth, convergence analysis and system characterization.

## Additional correction found during validation

Real image datasets expose class labels as their second item. Autoencoder workloads require the input image itself as the reconstruction target. v0.9.0 corrects `DatasetManager.sample_training_batch()` so `reconstruction`, `anomaly_detection` and `image_denoising` workloads use the input tensor as the target instead of an unrelated class label.
