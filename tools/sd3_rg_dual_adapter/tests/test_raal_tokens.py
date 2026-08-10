from __future__ import annotations

from raal_fakes import OffsetTokenizer

from sd3_rgda.cache import _prompt_for_classes, prompt_for_classes
from sd3_rgda.raal_tokens import DefectTextMaskBank, build_defect_text_mask


def test_prompt_wrapper_exact_equivalence() -> None:
    for bits in range(16):
        classes = tuple(index for index in range(4) if bits & (1 << index))
        assert prompt_for_classes(classes) == _prompt_for_classes(classes)


def test_t5_defect_mask_single_class() -> None:
    mask = build_defect_text_mask(OffsetTokenizer(), "road damage: longitudinal crack", (0,), 4)
    assert mask.shape == (1, 260)
    assert mask[:, :4].sum() == 0
    assert mask[:, 4:].sum() > 0


def test_t5_defect_mask_multiclass() -> None:
    prompt = prompt_for_classes((0, 2, 3))
    mask = build_defect_text_mask(OffsetTokenizer(), prompt, (0, 2, 3), 6)
    assert mask[:, 6:].sum() >= 5


def test_t5_negative_mask_zero() -> None:
    mask = build_defect_text_mask(OffsetTokenizer(), prompt_for_classes(()), (), 5)
    assert int(mask.sum().item()) == 0


def test_t5_mask_combined_length() -> None:
    bank = DefectTextMaskBank.build(OffsetTokenizer(), 9)
    assert len(bank.masks) == 16
    assert bank.lookup((1, 3)).shape == (1, 265)
    assert len(bank.sha256()) == 64
