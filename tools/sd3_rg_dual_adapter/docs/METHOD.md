# Method

SD3-RGDA adds zero-initialized residual adapters after SD3 image patch embedding. Normal road context tokens come from pseudo-clean road latents, while defect tokens come from RDD2022 box/class RG maps.

The residual is:

`H = H0 + gn(t) * Anormal(N) + gd(t) * Mtoken * Adefect(R)`

All newly added residual outputs are zero at initialization so an untrained wrapper is numerically equivalent to the base mock transformer within `atol <= 1e-6`.
