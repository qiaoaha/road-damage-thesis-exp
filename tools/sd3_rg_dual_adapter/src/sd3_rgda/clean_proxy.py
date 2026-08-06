"""Auditable pseudo-clean road condition construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from sd3_rgda.conditions import Box, build_rg_map


@dataclass(frozen=True)
class CleanProxyAudit:
    method: str
    output_size: tuple[int, int]
    mask_pixels: int
    outside_mask_unchanged: bool
    negative_exact_match: bool
    positive_mask_changed: bool
    gray_rectangle_method_used: bool = False
    sam_used: bool = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_box_mask(boxes: list[Box] | tuple[Box, ...], image_size: tuple[int, int], dilation: int = 8) -> Image.Image:
    height, width = image_size
    mask = Image.new("L", (width, height), 0)
    pixels = mask.load()
    if pixels is None:
        raise RuntimeError("Could not access mask pixels")
    for box in boxes:
        if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
            raise ValueError(f"Out-of-bounds box for clean proxy: {box}")
        x1 = max(0, box.x1 - dilation)
        y1 = max(0, box.y1 - dilation)
        x2 = min(width, box.x2 + dilation)
        y2 = min(height, box.y2 + dilation)
        for y in range(y1, y2):
            for x in range(x1, x2):
                pixels[x, y] = 255
    return mask


def telea_inpaint_proxy(image_path: str | Path, boxes: list[Box] | tuple[Box, ...]) -> tuple[Image.Image, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    if not boxes:
        return image.copy(), Image.new("L", image.size, 0)
    mask = build_box_mask(boxes, (image.height, image.width))
    import cv2

    arr = np.asarray(image)
    mask_arr = np.asarray(mask)
    out = cv2.inpaint(arr, mask_arr, 5, cv2.INPAINT_TELEA)
    proxy = Image.fromarray(out).convert("RGB")
    # OpenCV may alter tiny numerical details outside the mask. The contract is
    # exact preservation, so paste original pixels outside the audited region.
    proxy = Image.composite(proxy, image, mask)
    return proxy, mask


def audit_clean_proxy(original: Image.Image, proxy: Image.Image, mask: Image.Image, is_negative: bool) -> CleanProxyAudit:
    original_arr = np.asarray(original.convert("RGB"))
    proxy_arr = np.asarray(proxy.convert("RGB"))
    mask_arr = np.asarray(mask.convert("L")) > 0
    outside = ~mask_arr
    outside_unchanged = bool(np.array_equal(original_arr[outside], proxy_arr[outside]))
    exact_match = bool(np.array_equal(original_arr, proxy_arr))
    changed_inside = bool(mask_arr.any() and np.any(original_arr[mask_arr] != proxy_arr[mask_arr]))
    return CleanProxyAudit(
        method="OPENCV_TELEA_INPAINT",
        output_size=proxy.size,
        mask_pixels=int(mask_arr.sum()),
        outside_mask_unchanged=outside_unchanged,
        negative_exact_match=(exact_match if is_negative else True),
        positive_mask_changed=(changed_inside if not is_negative else True),
    )


def build_clean_pair(image_path: str | Path, boxes: list[Box] | tuple[Box, ...]) -> tuple[Image.Image, Image.Image, object]:
    target = Image.open(image_path).convert("RGB")
    clean, _mask = telea_inpaint_proxy(image_path, boxes)
    rg_map = build_rg_map(boxes, (target.height, target.width))
    return clean, target, rg_map
