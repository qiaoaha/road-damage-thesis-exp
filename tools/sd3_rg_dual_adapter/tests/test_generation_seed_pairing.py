from sd3_rgda.generation_manifest import derive_generation_seed


def test_generation_seed_is_per_index_and_shared_by_pair() -> None:
    assert derive_generation_seed(2026, 0) == 202600000
    assert derive_generation_seed(2026, 999) == 202600999
    assert len({derive_generation_seed(2026, index) for index in range(1000)}) == 1000
