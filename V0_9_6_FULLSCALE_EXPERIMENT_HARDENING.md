# v0.9.6 Full Scale Experiment Hardening

This release completes the final safeguards required before the hierarchical fingerprinting full scale campaign.

The server now has two explicit federated experiment modes. `full_scale` defaults to 100 global rounds and rejects values below 100. `smoke_test` accepts 1 through 99 rounds for debugging. The client still receives the policy from the server and does not choose the global round count.

Autoencoder anomaly detection now uses a genuinely disjoint calibration protocol. A deterministic normal only validation subset is reserved from the training split before IID or non IID client partitioning. Those calibration samples are removed from every client training shard. The anomaly threshold is recalibrated from that fixed normal validation subset each round and is never tuned on the test set.

Multiclass classification metrics now expose explicit `macro_precision`, `macro_recall`, and `macro_f1` aliases while preserving legacy metric columns. Anomaly detection continues to use positive class binary precision, recall, and F1 together with AUROC, AUPRC, the threshold, and the confusion counts.

The proxy now performs a disk space preflight and uses dumpcap or tshark ring buffer rotation. The default chunk target is 2048 MB. Rotated PCAPs are treated as one logical capture, hashed, inventoried, and processed one chunk at a time. `capture_chunks_manifest.json` records the number of chunks, total capture bytes, total packets, feature row count, extraction mode, and capture completion state.

Sparse feature windows are retained. Every feature row records `packet_information_threshold` and `packet_information_ok`; both are quality metadata and are explicitly excluded from the predictor matrix.

The server writes `round_progress.json` after every successful aggregation. Model checkpoints are written periodically according to `checkpoint.interval_rounds`, with a rolling latest checkpoint and a bounded number of archived checkpoints. Per round CSV writes are flushed and fsynced immediately.

Role status files use the terminal states `COMPLETED`, `PARTIAL`, `FAILED`, `CAPTURE_INCOMPLETE`, and `METRICS_INCOMPLETE`. Generic runner cleanup no longer overwrites a more specific integrity failure. `validate_experiment_run.py` performs a final cross role check of server rounds, server metrics, client update metrics, client role completion, proxy completion, PCAP chunks, capture bytes, and extracted packet counts and writes a run level `experiment_status.json`.

Each role now writes `reproducibility.json` containing execution, model initialization, shuffle, partition and anomaly calibration seeds together with Python, OS, hardware label, installed framework package versions, CUDA, cuDNN, and GPU information where available.

The release test suite contains 135 passing tests.
