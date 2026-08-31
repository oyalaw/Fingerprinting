# v0.9.8 Transformer application fingerprinting

Version 0.9.8 removes the deterministic application-label limitation from the
native Transformer branch and extends the fingerprint classifier to infer:

`family -> architecture -> variant -> application`

## Native Transformer applications

Tiny Transformer, BERT, and DistilBERT now expose both:

- `text_classification`
- `masked_language_modeling`

Masked language modeling uses the existing text corpora without requiring
class labels as training targets. Token 0 is padding, token 1 is reserved as
the experiment mask token, and unmasked target positions use `-100` so loss
and accuracy are computed only on masked tokens. The default masking
probability is 0.15 and is controlled by
`masked_language_modeling.mask_probability`.

ViT remains `image_classification` and DETR remains `object_detection` because
additional task heads are not yet implemented by the native runner.
Question answering and token classification are intentionally not exposed as
native applications until structured target datasets and metrics are added.

## Application-level fingerprinting

Prepared ground truth already contains `application`; v0.9.8 now trains a
fourth conditional classifier under each concrete variant. Application-stage
metrics use the same experiment-grouped cross-validation and Fisher-score
feature selection rules as family, architecture, and variant stages.

Legacy prepared datasets that predate the application field remain readable;
they receive an internal `__unknown_application__` constant label and therefore
do not provide application-level evaluation evidence.
