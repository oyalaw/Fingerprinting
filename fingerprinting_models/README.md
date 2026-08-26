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

## v0.8.7 note

Model bundle schema 1.1 uses stage-specific top-10 Fisher-score feature
selection. Bundles trained by v0.8.6 schema 1.0 should be retrained with:

```bash
python prepare_fingerprinting_dataset.py
python train_fingerprinting_models.py
```

The trainer writes `hierarchical_metrics.csv` with grouped Accuracy,
Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, and Log Loss when
there are enough independent experiment groups per class.
