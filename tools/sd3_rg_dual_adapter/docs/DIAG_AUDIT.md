# DIAG Audit

- Upstream repository: https://github.com/intelligolabs/DIAG
- Local reuse mode: git submodule only.
- Observed generation entry: `generate_augmented_images.py`.
- Observed model family: DIAG README references SDXL/diffusers and Stability AI SDXL assets.
- Reused concepts: prompt descriptions, region mask, road background condition, fixed seed, per-image generation metadata, and manifest separation.
- Not reused: DIAG source code, environment pinning, SDXL pipeline code, or generated assets.

`DIAG_UPSTREAM_COMMIT` is recorded in `reports/00_IMPLEMENTATION_STATUS.md`.
