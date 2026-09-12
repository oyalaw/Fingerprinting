# Safe packet cross-vantage replacement

These are drop-in replacements for:

- `train_packet_cross_vantage.py`
- `run_packet_cross_vantage_fingerprinting.py`

The existing 3.2 GB packet dataset is reused by default. The runner no longer
rebuilds it unless a requested packet-budget CSV is missing or `--rebuild` is
explicitly supplied.

## Install

From `~/lawrence/Fingerprinting`:

```bash
cp train_packet_cross_vantage.py train_packet_cross_vantage.py.before_safe_sampling
cp run_packet_cross_vantage_fingerprinting.py run_packet_cross_vantage_fingerprinting.py.before_safe_sampling
```

Copy the replacement files into the repository root, then syntax-check:

```bash
python3 -m py_compile train_packet_cross_vantage.py
python3 -m py_compile run_packet_cross_vantage_fingerprinting.py
```

## Recommended first validation run

```bash
python3 run_packet_cross_vantage_fingerprinting.py \
  --packet-counts 1,100 \
  --max-samples-per-trace 2000
```

This keeps all 26 experiments and all 78 proxy client traces while retaining at
most 2,000 packet/block observations from each trace per packet budget.

## Complete experiment

```bash
python3 run_packet_cross_vantage_fingerprinting.py \
  --packet-counts 1,5,10,25,50,100,250,500 \
  --max-samples-per-trace 5000
```

## New outputs

Sampling audit:

`fingerprinting_results/packet_cross_vantage/_sampling_audit/`

For every evaluated hierarchy stage:

- packet/block OOF metrics and confusion matrices
- `oof_predictions.csv`
- `selected_features_by_fold.json`
- `client_trace_fusion/metrics.json`
- `client_trace_fusion/confusion_matrix.{png,pdf}`
- `client_trace_fusion/oof_fused_predictions.csv`

The packet-level matrix N is the number of selected packet/block prediction
units. The fusion matrix N is the number of complete client traces.

## Scientific safeguards

- sampling is balanced per `(experiment_id, client_id, source_role)`
- deterministic stratified sampling spans the entire trace
- sampling is label-blind
- experiment IDs are disjoint between train/test
- Fisher ranking is recomputed inside every training fold
- packet fusion averages only held-out OOF probabilities
- proxy->proxy historical evaluation is explicitly labelled proxy->proxy
