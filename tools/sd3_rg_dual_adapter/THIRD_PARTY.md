# Third-Party Notes

- DIAG: external research reference and Git submodule at `third_party/DIAG`. The upstream repository is not copied into local clean-room source files.
- RoadFusion: paper-described method used as a conceptual reference. No official implementation was available in this task scope.
- Hugging Face Diffusers SD3 LoRA example: API reference only. The current upstream example targets newer development diffusers, so this scaffold does not copy its training loop.
- PyTorch, NumPy, Pillow, PyYAML: runtime/development dependencies for CPU/mock implementation.
