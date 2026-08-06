from __future__ import annotations

import types
from pathlib import Path

import numpy as np
from PIL import Image

from sd3_rgda.clean_proxy import audit_clean_proxy, telea_inpaint_proxy
from sd3_rgda.conditions import Box


def test_clean_proxy_preserves_outside_mask_and_negative_exact(tmp_path: Path, monkeypatch: object) -> None:
    def fake_inpaint(arr: np.ndarray, mask: np.ndarray, radius: int, method: int) -> np.ndarray:
        out = arr.copy()
        out[mask > 0] = np.array([9, 8, 7], dtype=np.uint8)
        return out

    import sys

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(inpaint=fake_inpaint, INPAINT_TELEA=1))  # type: ignore[attr-defined]
    image_path = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(image_path)
    proxy, mask = telea_inpaint_proxy(image_path, [Box(10, 10, 16, 16, 0)])
    audit = audit_clean_proxy(Image.open(image_path), proxy, mask, is_negative=False)
    assert audit.method == "OPENCV_TELEA_INPAINT"
    assert audit.outside_mask_unchanged
    assert audit.positive_mask_changed
    neg_proxy, neg_mask = telea_inpaint_proxy(image_path, [])
    neg = audit_clean_proxy(Image.open(image_path), neg_proxy, neg_mask, is_negative=True)
    assert neg.negative_exact_match
