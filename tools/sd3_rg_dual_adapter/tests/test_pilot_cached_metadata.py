from __future__ import annotations

import torch

from sd3_rgda.cache import CachedSD3Sample, load_cached_sample


def test_pilot_cached_metadata_round_trip(tmp_path) -> None:
    sample = CachedSD3Sample(
        image_path="image.png",
        label_path="label.txt",
        target_latent=torch.zeros(1, 1, 2, 2),
        pseudo_clean_latent=torch.zeros(1, 1, 2, 2),
        prompt_embeds=torch.zeros(1, 1, 2),
        pooled_prompt_embeds=torch.zeros(1, 2),
        rg_map_latent=torch.zeros(1, 7, 2, 2),
        token_mask=torch.zeros(1, 1, 1),
        class_ids=(0,),
        is_negative=False,
        sample_id="pilot_train_0001",
        anchor_class="D00",
        source_image_sha256="source",
        clean_proxy_sha256="clean",
        split="train",
    )
    path = tmp_path / "sample.pt"
    torch.save(sample.__dict__, path)
    loaded = load_cached_sample(path, torch.device("cpu"), torch.float32)
    assert loaded.sample_id == "pilot_train_0001"
    assert loaded.anchor_class == "D00"
    assert loaded.source_image_sha256 == "source"
    assert loaded.clean_proxy_sha256 == "clean"
