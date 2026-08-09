# SD3-RGDA Generation1000 And YOLO Ablation Plan

This stage prepares code only. It does not enable GPU, load real SD3, generate images, run YOLO, modify Formal5000 training logic, or commit model files.

## Purpose

The experiment asks whether the Formal5000 SD3-RGDA adapter improves downstream Czech road-damage detection when it is the only changed generation variable.

## Fixed Model

The fixed base code is `eb526b07553260d74f0db67a59505bc149fc241f`. The only permitted RGDA inference checkpoint is `best_eval.pt` from Formal5000 step 4250 with SHA256 `dca7d04a8b90b317724bd3572b21b91304f9c98fa36014208cc695df34489bc2`. The checkpoint is adapter-only and must not be committed to GitHub.

## Baseline Trace

The old Czech SD3-1000 baseline is reusable only if the exact 1000 sources, prompts, seeds, inference parameters, scheduler, labels, and output mapping are recovered. If any part remains unproven, the GPU phase regenerates a paired baseline:

| Group | Generator | RGDA | Source | Seed |
|---|---|---|---|---|
| G0 | SD3 Medium | No | Generation1000 row | row seed |
| G1 | SD3 Medium + best_eval.pt | Yes | same row | same seed |

## Source Selection

Generation1000 uses only Czech train sources. The preferred deterministic design is all 750 positive train images plus 250 negatives sampled from the 1230 train negatives with seed 2026. Val and test are never used for generation or training augmentation.

## Prompt And Seed

Prompts are deterministic class templates, with no aesthetic style words. Seeds are per-row:

```text
seed_i = 202600000 + generation_index
```

G0 and G1 recreate a fresh generator with the same seed for each paired image so the initial noise is identical.

## Inference Mechanism

The generation entrypoint uses Diffusers `StableDiffusion3Pipeline` in the GPU phase. It must preserve the official scheduler, timesteps, latent update, and VAE decode. RGDA is injected only around the SD3 transformer patch embedding by a temporary context manager. The context restores `transformer.forward` in `finally`, even when generation raises.

For RGDA mode, every SD3 transformer forward extracts the current timestep and builds an `RGDAConditionBatch` with cached pseudo-clean latent, RG map latent, token mask, and timestep. `RGDAPatchHook` is registered for that forward and must fire exactly once. Base mode never enters RGDA and has zero injector calls.

## Cache And Guidance

The formal train cache is the preferred source for pseudo-clean latents, prompt embeddings, pooled prompt embeddings, RG map latents, token masks, class ids, anchor class, and negative flag. If recovered baseline guidance is greater than 1, negative prompt embeddings must be supplied correctly; guidance must not be silently changed.

## Labels

No pseudo-labeling is allowed. Each generated image receives a direct copy of the source YOLO label, and SHA256 of copied label must equal SHA256 of the source label. Negative source labels remain empty or exactly identical.

## Output

The formal GPU output root is:

```text
/root/autodl-tmp/road_damage_exp/generated/sd3_rgda_generation1000/
```

It contains `paired_manifest.csv`, `base/images`, `base/labels`, `rgda/images`, `rgda/labels`, and `qa` reports. If G0 is fully reused from an old baseline, the base directory may be represented by an external-reference manifest instead of copying images.

## Resume And QA

Resume is valid only when image, label, image dimensions, image SHA, label SHA, mode, seed, and checkpoint SHA match the results manifest. QA gates include successful rows, unique files and SHAs, valid RGB images, label SHA match, negative label gate, val/test leakage, checkpoint SHA match, and paired source/seed/prompt equality.

## YOLO Ablation

Datasets use fixed splits:

| Group | Train data | Synthetic | RGDA | P | R | mAP50 | mAP50-95 |
|---|---|---:|---|---:|---:|---:|---:|
| D0 | Real1980 | 0 | No | | | | |
| D1 | Real1980+SD3 | 1000 | No | | | | |
| D2 | Real1980+SD3-RGDA | 1000 | Yes | | | | |

Val and test are identical real Czech splits in all groups. Synthetic images are train-only and prefixed as `sd3base_` or `sd3rgda_` to avoid collisions.

Primary deltas are:

```text
Delta SD3 = D1 - D0
Delta RGDA = D2 - D1
Delta Total = D2 - D0
```

The paper-relevant quantities are `Delta RGDA mAP50` and `Delta RGDA mAP50-95`.

## GPU Execution Design

Future GPU execution is staged as environment gate, 16-case smoke, Generation1000, QA, YOLO dataset build, YOLO seed 2026 training, result summary, packaging, download, SHA verification, and shutdown. The 16-case smoke includes 2 each for D00/D10/D20/D40 and 8 negatives, paired if G0 must be regenerated.

## Failure Handling

Checkpoint SHA mismatch, cache join ambiguity, source leakage, label SHA mismatch, corrupted images, seed mismatch, and synthetic val/test leakage are hard failures. OOM may be retried once only in the GPU phase with explicit evidence packaging. No thresholds or generation parameters may be changed silently.

## Thesis Use

The final chapter reports whether adding RGDA at generation time gives a cleaner label-preserving synthetic augmentation than SD3 alone under the same source, prompt, seed, scheduler, resolution, steps, guidance, dtype, labels, and YOLO training protocol.
