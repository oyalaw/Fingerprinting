# Fingerprinting model bundles

This directory intentionally contains no pretrained publication model.

Run:

```bash
python3 prepare_fingerprinting_dataset.py
python3 train_fingerprinting_models.py
```

to create model bundles from your own proxy-only observations and
ground-truth labels.

The current ResNet101/MobileNetV2 pilot data is useful for validating
architecture separability, but one independent run per architecture is
not enough for defensible cross-run performance estimates. The trainer
will mark such evaluations `insufficient_independent_runs`.

A bundle may still be used as an engineering/pipeline smoke test, but
it should not be presented as a publication-quality classifier until
independent-run evaluation is available.


## v0.8.7 Fisher outputs

Every trained bundle now includes `fisher_scores.csv`, and `metadata.json`
contains the complete stage-specific Fisher ranking and selected feature list.

Model bundle schema 1.1 is not compatible with v0.8.6 schema-1.0 bundles.
Retrain models after upgrading.
