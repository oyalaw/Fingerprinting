# v0.8.7 Hierarchy and Fingerprinting Corrections

This update is based on the v0.8.6 dual-inference codebase and corrects the
hierarchy/availability problems observed during the Transformer and
Autoencoder experiments.

## Corrections implemented

1. **Transformer hierarchy no longer collapses to one child.**
   - Tiny Transformer: 2-layer, 4-layer, 6-layer (PyTorch + TensorFlow).
   - BERT: Tiny, Mini, Small, Base, Large (PyTorch; Hugging Face `transformers`
     required; instantiated from config with random weights, so no pretrained
     model download is required).
   - DistilBERT Base (PyTorch + `transformers`).
   - ViT B/16, B/32, L/16 (PyTorch `torchvision`).
   - DETR remains catalogued as artifact-only because the generic
     classification training loop does not implement detection targets/losses.

2. **Autoencoder hierarchy is genuinely hierarchical.**
   - Dense Autoencoder: 2-layer, 3-layer, 5-layer.
   - Convolutional Autoencoder: 2-layer, 4-layer, 6-layer.
   - Variational Autoencoder: FC-VAE, Conv-VAE, Beta-VAE.
   - Implemented in PyTorch and TensorFlow. VAE training includes the KL term;
     Beta-VAE uses beta=4.

3. **MLP family is executable instead of taxonomy-only.**
   - Feedforward MLP: 2-layer, 4-layer, 8-layer.
   - PyTorch + TensorFlow.

4. **Catalog-only models are no longer silently hidden conceptually.**
   - `python main.py models --framework pytorch` prints the complete taxonomy
     with `native` versus `artifact-only` status.
   - During native interactive configuration the program prints which
     families/architectures/variants are catalog-only and therefore not
     selectable for native training.
   - GAN, diffusion, current GNN/state-space entries, and detection/segmentation
     models remain artifact-only until their task-specific training loops are
     implemented correctly. They are not mislabeled as generic native models.

5. **Legacy configuration migration is complete.**
   - Every concrete registered variant now maps automatically from the older
     `ai.architecture=<concrete model>` format.
   - This includes prior experimental variants such as `resnet101`,
     `mobilenet_v3_large`, and `efficientnet_b2`.

6. **Device/OS metadata separation is enforced.**
   - Added `dell_desktop`, `dell_laptop`, `generic_linux_desktop`, and
     `generic_windows_desktop` device labels.
   - Values such as `ubuntu`, `linux`, or `windows` are rejected as
     `device.label`; they belong in `device.operating_system`.

7. **Fisher-score feature selection is now in the actual hierarchical trainer.**
   - Each non-constant Family/Architecture/Variant stage computes a multiclass
     Fisher score over proxy-only predictors.
   - Top 10 Fisher-ranked features are selected independently for that stage.
   - Random Forest is trained only on the selected features.
   - The bundle stores the Fisher ranking and selected feature list.

8. **Hierarchical metrics are expanded.**
   - Grouped evaluation now reports accuracy, balanced accuracy, macro
     precision, macro recall, macro F1, and log loss when at least two
     independent experiment groups exist per class.
   - `train_fingerprinting_models.py` writes
     `fingerprinting_models/hierarchical_metrics.csv`.

9. **Accidental 5-second-only real-time runs are fail-closed.**
   - Standard proxy windows remain `[0.5, 1.0, 2.0, 5.0]`.
   - If architecture inference is enabled and only one scale is supplied, the
     run is rejected unless `capture.allow_single_scale=true` is explicitly
     set. This prevents an old persisted proxy configuration from silently
     defeating the multiscale experiment.

## Installation

For the standard PyTorch hierarchy:

```bash
python -m pip install -e .
python -m pip install torch torchvision
```

For BERT/DistilBERT options:

```bash
python -m pip install transformers
```

Or install the project PyTorch extra:

```bash
python -m pip install -e '.[pytorch]'
```

## Validation performed

- `python -m pytest -q`: **77 passed**.
- `compileall`: run before packaging.
- PyTorch one-batch training smoke tests passed for the newly implemented
  Tiny Transformer, Dense/Convolutional/VAE Autoencoders, and MLP variants.
- ViT-B/16 and ViT-B/32 were instantiated and trained for one synthetic batch.
- BERT/DistilBERT builders compile but were not runtime-smoke-tested in the
  build container because the optional `transformers` package is not installed
  there.
- TensorFlow builders compile but were not runtime-smoke-tested in the build
  container because TensorFlow is not installed there.

## Research interpretation

A hierarchy level is only considered a fingerprinting discrimination problem
when it has competing children. Constant single-child stages are still stored
as `kind=constant`, but must not be reported as learned 100% classification.
