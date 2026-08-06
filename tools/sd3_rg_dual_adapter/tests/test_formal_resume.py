from __future__ import annotations

import pytest

from sd3_rgda.formal_engine import FORMAL_CONFIG, _validate_formal_config


def test_formal_resume_rejects_core_config_mismatch() -> None:
    config = {**FORMAL_CONFIG, "TRAIN_STEPS": 5000, "LEARNING_RATE": 2e-4}
    with pytest.raises(ValueError, match="FORMAL_RESUME_CORE_CONFIG_MISMATCH"):
        _validate_formal_config(config, 5000)


def test_formal_resume_accepts_matching_core_config() -> None:
    _validate_formal_config({**FORMAL_CONFIG, "TRAIN_STEPS": 5000}, 5000)
