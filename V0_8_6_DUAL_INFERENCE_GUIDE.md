# v0.8.6 Dual Inference Guide

## Goal

The proxy supports both:

1. **Progressive real-time inference** while traffic is still flowing.
2. **Final complete-trace inference** after the experiment stops.

Both use proxy-observable traffic only.

## Collection before models exist

You can run v0.8.6 before training any architecture model. The proxy will still
write:

```text
<run>_live_features.csv
```

at 0.5/1/2/5-second scales and the standard end-of-run per-client feature
files. It will state that no trained bundles were found rather than inventing
a prediction.

## Prepare labels and X/Y

Place matching proxy, client, and server results under the project tree, then:

```bash
python3 prepare_fingerprinting_dataset.py
```

The generated X contains network predictors only. Ground-truth family,
architecture, variant, device, and other labels remain in Y.

## Train

Run:

```bash
python3 train_fingerprinting_models.py
```

No arguments are required.

For every available scale, the script trains:

```text
full/final
full/realtime_<scale>
size_normalized/final
size_normalized/realtime_<scale>
```

A single-family dataset can still learn an architecture classifier inside that
family. A single variant under an architecture is recorded as a constant
candidate, not as evidence that variant-level discrimination has been learned.

## Run with trained models

Start the proxy normally and answer yes to:

```text
Enable architecture fingerprinting (real-time + end-of-experiment)
Enable real-time progressive architecture inference
Enable final complete-trace architecture inference
```

The live console prints predictions when model bundles are available.

## Research metrics

For real-time inference report:

```text
balanced accuracy
macro F1
confidence
time to first confident prediction
time to stable prediction
abstention rate
```

For final inference report:

```text
balanced accuracy
macro F1
confusion matrix
per-class precision/recall/F1
```

Compare both `full` and `size_normalized` modes.

## Leakage control

Never randomly split windows from one experiment between training and test.
The independent experimental run is the grouping unit. Use unseen runs, and
where possible unseen device/session combinations, for evaluation.
