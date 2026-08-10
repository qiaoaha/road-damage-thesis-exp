# SD3-RGDA RAAL Plan

BASE_COMMIT=d4750be67ca680db00213df8e2989d4a6dad592c

RAAL is Region Attention Alignment Loss: a CHIMERA-inspired attention-mask supervision branch for SD3-RGDA. It is not an exact CHIMERA loss reproduction.

The implementation supervises only the T5 prompt segment. CLIP token length is read from the prompt embedding contract and combined with T5 max length 256. Negative samples produce an all-zero defect text mask and RAAL loss is exactly zero.

The attention processor is not replaced. RAAL registers temporary `forward_pre_hook(with_kwargs=True)` hooks on selected `transformer.transformer_blocks[i].attn` layers and reads Q/K through `to_q` and `add_k_proj` in an auxiliary branch. The original attention output remains unchanged.

Pilot1000 has two arms sharing an identical source schedule:

- R0: SD3-RGDA with RAAL disabled and weight 0.0.
- R1: SD3-RGDA with RAAL enabled and weight 0.02.

The code-ready stage is local only: no server, no RTX 5090, no real SD3 load, no Pilot1000 training, no generation, and no YOLO execution.
