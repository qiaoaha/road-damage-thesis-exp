# Architecture

The implementation is organized around small, testable modules:

- `conditions.py`: seven-channel RDD2022 region guidance maps.
- `clean_proxy.py`: pseudo-clean input construction for positive samples and identity proxy for negative samples.
- `condition_encoders.py`: latent and RG map encoders aligned to SD3 image token dimensions.
- `adapters.py`: zero-initialized normal and defect residual adapters.
- `timestep_gate.py`: independent tanh gates initialized to zero.
- `sd3_hook.py` and `sd3_rg_transformer.py`: controlled hook/wrapper utilities.
- `roadfusion_reference.py`: clean-room dual-adapter reference head.
