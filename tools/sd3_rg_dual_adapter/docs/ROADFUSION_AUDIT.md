# RoadFusion Audit

- Paper: RoadFusion, arXiv:2507.15346.
- Audited concept: dual feature adapters for normal and anomalous representations plus a patch-level normality discriminator.
- Clean-room reference file: `src/sd3_rgda/roadfusion_reference.py`.
- SD3 adaptation: RoadFusion patch features are mapped to SD3 image tokens, and anomalous features are mapped to RDD2022 box/class region condition tokens.
- Training details in this scaffold are limited to the CPU/mock gate and do not claim official reproduction.
