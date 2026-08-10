from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any

import torch

from sd3_rgda.cache import CLASS_PHRASES, prompt_for_classes

T5_MAX_LENGTH = 256


def _offsets_from_tokenizer(tokenizer_3: Any, prompt: str, max_sequence_length: int) -> list[tuple[int, int]]:
    encoded = tokenizer_3(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"] if isinstance(encoded, dict) else encoded.offset_mapping
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets]


def _phrase_spans(prompt: str, class_ids: tuple[int, ...]) -> list[tuple[int, int]]:
    lowered = prompt.lower()
    spans: list[tuple[int, int]] = []
    for class_id in class_ids:
        phrase = CLASS_PHRASES[class_id]
        start = lowered.find(phrase)
        if start < 0:
            raise ValueError(f"Defect phrase not found in prompt: {phrase}")
        spans.append((start, start + len(phrase)))
    return spans


def build_defect_text_mask(
    tokenizer_3: Any,
    prompt: str,
    class_ids: tuple[int, ...] | list[int],
    clip_seq_len: int,
    max_sequence_length: int = T5_MAX_LENGTH,
) -> torch.Tensor:
    ids = tuple(class_ids)
    total_text_len = clip_seq_len + max_sequence_length
    mask = torch.zeros((1, total_text_len), dtype=torch.bool)
    if not ids:
        return mask
    spans = _phrase_spans(prompt, ids)
    offsets = _offsets_from_tokenizer(tokenizer_3, prompt, max_sequence_length)
    t5_mask = torch.zeros(max_sequence_length, dtype=torch.bool)
    for span_start, span_end in spans:
        before = int(t5_mask.sum().item())
        for token_index, (token_start, token_end) in enumerate(offsets[:max_sequence_length]):
            if token_end <= token_start:
                continue
            if token_start < span_end and token_end > span_start:
                t5_mask[token_index] = True
        if int(t5_mask.sum().item()) == before:
            raise ValueError(f"No tokenizer offsets matched prompt span {span_start}:{span_end}")
    mask[:, clip_seq_len:] = t5_mask
    return mask


@dataclass(frozen=True)
class DefectTextMaskBank:
    masks: dict[tuple[int, ...], torch.Tensor]
    clip_seq_len: int
    max_sequence_length: int = T5_MAX_LENGTH

    @classmethod
    def build(
        cls,
        tokenizer_3: Any,
        clip_seq_len: int,
        combos: list[tuple[int, ...]] | None = None,
        max_sequence_length: int = T5_MAX_LENGTH,
    ) -> DefectTextMaskBank:
        if combos is None:
            combos = []
            for length in range(5):
                combos.extend(tuple(combo) for combo in itertools.combinations(CLASS_PHRASES, length))
        masks = {
            tuple(combo): build_defect_text_mask(
                tokenizer_3,
                prompt_for_classes(tuple(combo)),
                tuple(combo),
                clip_seq_len,
                max_sequence_length,
            )
            for combo in combos
        }
        return cls(masks=masks, clip_seq_len=clip_seq_len, max_sequence_length=max_sequence_length)

    def lookup(self, class_ids: tuple[int, ...] | list[int]) -> torch.Tensor:
        key = tuple(class_ids)
        if key not in self.masks:
            raise KeyError(f"Missing RAAL defect text mask for classes {key}")
        return self.masks[key]

    def sha256(self) -> str:
        payload = {
            ",".join(str(item) for item in key): value.to(dtype=torch.uint8).cpu().tolist()
            for key, value in sorted(self.masks.items())
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
