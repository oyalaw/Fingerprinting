# v0.9.7 — VAE application taxonomy correction

The variational autoencoder branch previously exposed only `reconstruction` for `vae_fc`, `vae_conv`, and `beta_vae`, even though the native PyTorch and TensorFlow workload paths already support reconstruction-error based anomaly detection.

The corrected hierarchy is:

```text
autoencoder
  variational_autoencoder
    vae_fc
      reconstruction
      anomaly_detection
    vae_conv
      reconstruction
      anomaly_detection
    beta_vae
      reconstruction
      anomaly_detection
```

For anomaly detection, training remains reconstruction based. A held-out normal calibration subset determines the reconstruction-error threshold, and evaluation is performed on a separate fixed test set. Beta-VAE retains its beta-weighted KL training objective while anomaly scoring uses reconstruction MSE, consistent with the existing anomaly protocol.

`image_generation` is intentionally not exposed for native VAEs in this release because the current generic experiment runner does not implement a dedicated generation-quality evaluation protocol.
