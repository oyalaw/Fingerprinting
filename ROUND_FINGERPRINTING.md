# Round-aware fingerprinting extension

This extension adds **two additional fingerprinting options** and does not replace the existing complete-trace or real-time modes.

1. **Round-level fingerprinting**: one classifier decision per inferred FL round per client.
2. **Round-fusion fingerprinting**: average the out-of-fold class probabilities across all inferred rounds for one client trace, producing one final client-level decision.

## Important scientific rule

The proxy detector does **not** read the encrypted FL payload, true round number, model family, architecture, variant, dataset, or client/server training metrics. Round boundaries are inferred from proxy-observable directional traffic only:

`large DOWN transfer -> local-compute gap -> large UP transfer`

Endpoint `federated_phase` logs are used only to validate segmentation quality. They are never written into predictor X.

## Run

From the repository root:

```bash
python3 run_round_fingerprinting.py
```

Optional tuning:

```bash
python3 run_round_fingerprinting.py \
  --bin-sec 0.25 \
  --direction-dominance 0.65 \
  --min-major-transfer-bytes 262144 \
  --bridge-gap-sec 0.75
```

Build dataset without classifier evaluation:

```bash
python3 run_round_fingerprinting.py --skip-evaluation
```


## Complete-trace temporal fusion from realtime windows

The package also implements the previously discussed late-fusion option. It does not replace the realtime classifier. It uses experiment-disjoint OOF probabilities from each 0.5/1/2/5-second window and averages them per client trace:

```bash
python3 run_temporal_fusion.py
```

This produces one final prediction per client trace while preserving the short-window information that whole-trace statistical summarization can wash out. Outputs are written under `fingerprinting_results/temporal_fusion/`.

## Outputs

Dataset and segmentation audit:

- `fingerprinting_dataset/round_fingerprinting/round_X_proxy.csv`
- `fingerprinting_dataset/round_fingerprinting/round_Y_ground_truth.csv`
- `fingerprinting_dataset/round_fingerprinting/round_schema.json`
- `fingerprinting_dataset/round_fingerprinting/round_boundaries.csv`
- `fingerprinting_dataset/round_fingerprinting/round_inference_diagnostics.json`
- `fingerprinting_dataset/round_fingerprinting/round_boundary_validation.json`

Publication-ready grouped OOF evaluation:

- `fingerprinting_results/round_fingerprinting/full/round/...`
- `fingerprinting_results/round_fingerprinting/full/round_fusion/...`
- `fingerprinting_results/round_fingerprinting/size_normalized/round/...`
- `fingerprinting_results/round_fingerprinting/size_normalized/round_fusion/...`

Each evaluated stage includes metrics, confusion matrix CSV/PNG/PDF, ROC/PR curves when defined, OOF predictions, and fold-local Fisher feature selections.

## Evaluation leakage protection

All rounds belonging to the same `experiment_id` remain in the same fold. Fisher feature selection is recomputed using **training folds only**. Round-fusion uses only OOF probabilities, so the final fused client prediction is also experiment-disjoint.

## Interpretation of confusion-matrix N

- Round-level matrix: `N` is the number of inferred round predictions for that class.
- Round-fusion matrix: `N` is the number of complete client traces after fusing their round predictions.

The existing complete-trace and 0.5/1/2/5-second real-time datasets are untouched.
