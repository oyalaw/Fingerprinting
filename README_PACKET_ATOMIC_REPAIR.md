# Atomic compact packet dataset repair

The previous builder writes directly to `packet_K.csv` with mode `"w"`.
Therefore an interrupted rebuild can truncate a previously complete dataset
while leaving the older `build_summary.json` and inventory untouched.

This package fixes that failure mode and avoids materializing hundreds of
millions of rows that the classifier will later discard.

## Files

- `build_packet_cross_vantage_atomic_sampled.py`
- `run_packet_cross_vantage_fingerprinting_atomic.py`
- `validate_packet_cross_vantage_dataset.py`

The builder imports the existing `build_packet_cross_vantage_dataset.py` only
for its already-tested source discovery, canonical proxy identity resolution,
label loading, and packet sequence reconstruction.

## What changes

For each `(experiment_id, client_id, source_role)` trace and each packet budget:

- determine all available K-packet prediction units;
- select at most 5,000 units by default;
- selection is deterministic, label-blind, and stratified across the full trace;
- preserve exact original feature semantics;
- write to a staging directory;
- validate expected vs written traces/experiments;
- promote the staging directory only if validation passes;
- preserve the previous dataset as a timestamped backup by default.

If interrupted, the currently active dataset is not touched.

## Install

Copy the three Python files into `~/lawrence/Fingerprinting`.

Syntax check:

```bash
python3 -m py_compile build_packet_cross_vantage_atomic_sampled.py
python3 -m py_compile run_packet_cross_vantage_fingerprinting_atomic.py
python3 -m py_compile validate_packet_cross_vantage_dataset.py
```

## Repair the current partial packet dataset

Build only K=1 and K=100 first:

```bash
python3 build_packet_cross_vantage_atomic_sampled.py \
  --packet-counts 1,100 \
  --max-samples-per-trace 5000
```

The current partial directory is preserved as:

`fingerprinting_dataset/packet_cross_vantage.backup_<UTC timestamp>`

after the replacement validates successfully.

Validate:

```bash
python3 validate_packet_cross_vantage_dataset.py \
  --packet-counts 1,100
```

For the current corpus, K=1 should show 26 experiments and 78 proxy client
traces. K=100 should also normally show 26/78 because the discovered source
traces are much longer than 100 packets.

Then run the safe trainer already installed:

```bash
python3 run_packet_cross_vantage_fingerprinting_atomic.py \
  --packet-counts 1,100 \
  --max-samples-per-trace 2000
```

## Build all budgets

Once K=1 and K=100 validate:

```bash
python3 build_packet_cross_vantage_atomic_sampled.py \
  --packet-counts 1,5,10,25,50,100,250,500 \
  --max-samples-per-trace 5000
```

Then:

```bash
python3 run_packet_cross_vantage_fingerprinting_atomic.py \
  --packet-counts 1,5,10,25,50,100,250,500 \
  --max-samples-per-trace 5000
```

## Important scientific note

The `available_samples` field records how many exact packet/block prediction
units exist in the source traces. The `samples` field records how many were
actually materialized after trace-balanced sampling. Thus the source corpus can
still contain hundreds of millions of potential packet observations without
requiring a tens-of-GB analysis CSV.
