from __future__ import annotations

from scripts.build_formal_clean_proxy import formal_clean_proxy_qa_passed


def test_formal_clean_proxy_integer_gates() -> None:
    qa = {
        "PROXY_TOTAL": 2404,
        "PROXY_UNIQUE_SOURCE_IMAGES": 2404,
        "PROXY_MISSING": 0,
        "PROXY_CORRUPT": 0,
        "PROXY_SHAPE_MISMATCH": 0,
        "NEGATIVE_PROXY_EXACT_MATCH": "PASS",
        "POSITIVE_MASK_CHANGED": "PASS",
        "OUTSIDE_MASK_UNCHANGED": "PASS",
        "GRAY_RECTANGLE_METHOD_USED": "NO",
        "SAM_USED": "NO",
        "BLUR_FALLBACK_USED": "NO",
    }
    assert formal_clean_proxy_qa_passed(qa)
    qa["PROXY_CORRUPT"] = 1
    assert not formal_clean_proxy_qa_passed(qa)
