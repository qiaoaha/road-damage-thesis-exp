from pathlib import Path

from sd3_rgda.generation_manifest import sha256_file
from sd3_rgda.inference import copy_label_with_sha


def test_label_copy_preserves_sha(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    target = tmp_path / "out" / "target.txt"
    source.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    copied_sha = copy_label_with_sha(source, target)
    assert copied_sha == sha256_file(source)
    assert sha256_file(target) == sha256_file(source)
