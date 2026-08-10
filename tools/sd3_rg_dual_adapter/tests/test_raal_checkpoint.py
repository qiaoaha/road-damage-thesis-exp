from __future__ import annotations

import pytest

from sd3_rgda.raal import RAALConfig
from sd3_rgda.raal_engine import raal_checkpoint_metadata, validate_raal_resume_config


def test_raal_checkpoint_config_roundtrip() -> None:
    expected = raal_checkpoint_metadata(
        config=RAALConfig(),
        attention_mask_bank_sha256="a" * 64,
        source_schedule_sha256="b" * 64,
    )
    validate_raal_resume_config(dict(expected), expected)
    assert expected["raal_layers"] == "5,11,17"


def test_raal_resume_config_mismatch() -> None:
    expected = raal_checkpoint_metadata(
        config=RAALConfig(),
        attention_mask_bank_sha256="a" * 64,
        source_schedule_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="RAAL_RESUME_CONFIG_MISMATCH"):
        validate_raal_resume_config({**expected, "raal_weight": "0.0"}, expected)
