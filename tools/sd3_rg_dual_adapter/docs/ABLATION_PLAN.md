# Ablation Plan

- F0: original SD3, no RGDA.
- F1: SD3 plus one shared adapter.
- F2: SD3 plus Defect RG Adapter only.
- F3: SD3 plus Normal Adapter and Defect RG Adapter.
- F4: F3 plus timestep gate.
- F5: F4 plus optional LoRA.

Downstream detection:

- B0: Czech real training set, mAP50 = 0.2230.
- B1: Czech real + completed SD3 background-paste 1000, mAP50 = 0.2580.
- B2: Czech real + SD3-RGDA generated 1000.

Primary comparison: B2 vs B1. Suggested value gate: B2 mAP50 at least 0.2680, adding at least one percentage point over B1.
