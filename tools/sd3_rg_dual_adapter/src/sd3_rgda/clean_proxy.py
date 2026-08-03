"""Pseudo-clean road condition construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from sd3_rgda.conditions import Box, build_rg_map


def build_box_mask(boxes: list[Box] | tuple[Box, ...], image_size: tuple[int, int], dilation: int = 3) -> Image.Image:
    height, width = image_size
    mask = Image.new("L", (width, height), 0)
    pixels = mask.load()
    if pixels is None:
        raise RuntimeError("Could not access mask pixels")
    for box in boxes:
        if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
            raise ValueError(f"Out-of-bounds box for clean proxy: {box}")
        for y in range(box.y1, box.y2):
            for x in range(box.x1, box.x2):
                pixels[x, y] = 255
    if dilation > 0:
        mask = mask.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
    return mask


def telea_or_blur_proxy(image_path: str | Path, boxes: list[Box] | tuple[Box, ...]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if not boxes:
        return image.copy()
    mask = build_box_mask(boxes, (image.height, image.width))
    try:
        import cv2

        arr = np.asarray(image)
        mask_arr = np.asarray(mask)
        out = cv2.inpaint(arr, mask_arr, 3, cv2.INPAINT_TELEA)
        return Image.fromarray(out)
    except ImportError:
        blurred = image.filter(ImageFilter.GaussianBlur(radius=7))
        return Image.composite(blurred, image, mask)


def build_clean_pair(image_path: str | Path, boxes: list[Box] | tuple[Box, ...]) -> tuple[Image.Image, Image.Image, object]:
    target = Image.open(image_path).convert("RGB")
    clean = telea_or_blur_proxy(image_path, boxes)
    rg_map = build_rg_map(boxes, (target.height, target.width))
    return clean, target, rg_map
