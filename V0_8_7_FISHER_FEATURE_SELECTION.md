# v0.8.7 Stage-Specific Fisher Feature Selection

## Purpose

The hierarchical fingerprinting pipeline now learns different feature subsets
for:

```text
Family
Architecture | Family
Variant | Family, Architecture
```

The design is motivated by the observation that features separating
CNN-versus-Autoencoder need not be the same features separating
ResNet-versus-MobileNet, and variant-level discrimination may require another
subset again.

## Fisher score

For feature `j`, v0.8.7 uses a class-balanced multiclass Fisher score:

```text
            sum_c (mu_cj - mean_c(mu_cj))^2
F_j = ------------------------------------------------
            sum_c variance_cj + epsilon
```

Every class contributes equally. A class with many more 5-second windows does
not receive greater weight merely because its experiment lasted longer.

A higher Fisher score means stronger *univariate separability*. It does not
prove network-condition invariance.

## Selection defaults

```text
top_k       = 25
min_score   = 1e-6
min_features = 8
```

The trainer first keeps features above `min_score`, capped at `top_k`. If fewer
than `min_features` survive, the highest-ranked features are retained until the
minimum is reached.

The defaults are defined in:

```text
ai_fingerprint/architecture_models.py
```

## Stage-specific selection

The full hierarchy uses:

```text
F_family

F_architecture[family]

F_variant[family, architecture]
```

For example:

```text
F_family
    -> CNN vs Autoencoder

F_architecture[CNN]
    -> ResNet vs MobileNet

F_variant[CNN, ResNet]
    -> ResNet18 vs ResNet50 vs ResNet101
```

These sets are allowed to differ.

If a stage has only one represented class, that stage is stored as a constant
candidate and no Fisher selection is performed. This avoids reporting feature
importance for a classification problem that does not yet exist.

## Leakage control

For the final trained model, Fisher ranking is computed from all training data.

For reported grouped cross-validation metrics, however, feature selection is
performed *inside each training fold only*:

```text
split by experiment
        |
        +-- training experiments
        |       |
        |       +-- Fisher ranking
        |       +-- select features
        |       +-- fit Random Forest
        |
        +-- held-out experiments
                |
                +-- evaluate using the training-fold feature set
```

The held-out experiment is never used to choose features.

This is necessary because selecting features once on the full dataset before
cross-validation would leak information from the test runs.

## Full and size-normalized modes

Stage-specific Fisher selection operates independently for both:

```text
full
size_normalized
```

`full` includes all permitted proxy-observable fingerprint features.

`size_normalized` first removes the dominant absolute traffic-footprint fields
such as total bytes, packet counts, burst counts, and absolute TCP counts.
Fisher selection then ranks the remaining structural/rate/distributional
features.

This allows the study to distinguish:

```text
realistic architecture leakage
```

from:

```text
architecture information that survives suppression of model-size footprint
```

## Training

Prepare the X/Y dataset:

```bash
python3 prepare_fingerprinting_dataset.py
```

Then run:

```bash
python3 train_fingerprinting_models.py
```

No arguments are required.

For every trained bundle, v0.8.7 writes:

```text
bundle.pkl
metadata.json
fisher_scores.csv
```

Example directory:

```text
fingerprinting_models/
  full/
    final/
      bundle.pkl
      metadata.json
      fisher_scores.csv
```

The CSV fields are:

```text
mode
feature_mode
window_size_sec
stage
parent
rank
feature
fisher_score
selected
```

For architecture rows, `parent` is the family. For variant rows, `parent` is
`family::architecture`.

## What to report in the paper

Do not report a single global Fisher ranking for the entire hierarchy.

Report separate tables such as:

```text
Top Family Features
Top Architecture Features | CNN
Top Variant Features | CNN -> ResNet
```

and repeat the analysis for both full and size-normalized fingerprints where
space permits.

Fisher score is a feature-screening/separability statistic. Final performance
claims should still come from independent-run grouped evaluation, preferably
with additional leave-one-device-out and leave-one-network-condition-out
experiments.
