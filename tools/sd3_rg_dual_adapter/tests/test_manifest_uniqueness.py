from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_czech_manifest import choose_manifests


def _pair(name: str, classes: list[int]) -> tuple[Path, Path, list[int]]:
    return Path(f"{name}.jpg"), Path(f"{name}.txt"), classes


def test_choose_manifests_prefers_unique_single_class_samples() -> None:
    pairs = [
        _pair("multi_0_1", [0, 1]),
        _pair("d00_a", [0]),
        _pair("d00_b", [0]),
        _pair("d10_a", [1]),
        _pair("d10_b", [1]),
        _pair("d20_a", [2]),
        _pair("d20_b", [2]),
        _pair("d40_a", [3]),
        _pair("d40_b", [3]),
        _pair("negative", []),
    ]
    manifests = choose_manifests(pairs)
    smoke = manifests["gpu_smoke8.csv"]
    micro = manifests["micro_overfit4.csv"]
    negative = manifests["negative_smoke1.csv"]
    assert len(smoke) == 8
    assert len({row[0] for row in smoke}) == 8
    assert len(micro) == 4
    assert len({row[0] for row in micro}) == 4
    assert [row[2][0] for row in micro] == [0, 1, 2, 3]
    assert len({row[0] for row in negative}) == 1


def test_choose_manifests_fails_when_unique_samples_are_unavailable() -> None:
    pairs = [
        _pair("multi_0_1", [0, 1]),
        _pair("d20_a", [2]),
        _pair("d20_b", [2]),
        _pair("d40_a", [3]),
        _pair("d40_b", [3]),
        _pair("negative", []),
    ]
    with pytest.raises(RuntimeError, match="Cannot select"):
        choose_manifests(pairs)
