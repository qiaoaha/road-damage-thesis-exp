# Reproducibility

- Fixed Czech split: train 1980, val 424, test 425.
- Generated images inherit original labels only and must be training-only.
- CPU/mock gates must pass before GPU smoke.
- GPU smoke must use local files only and must not download SD3.
- Formal generation must preserve source image, mask, prompt, seed, class, parameters, labels, and manifest rows.
