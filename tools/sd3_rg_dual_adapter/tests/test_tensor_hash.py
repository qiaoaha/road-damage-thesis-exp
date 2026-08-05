from __future__ import annotations

import torch

from sd3_rgda.real_sd3_engine import hash_tensor_bytes


def test_hash_tensor_bytes_is_stable_for_float_dtypes() -> None:
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        tensor = torch.arange(4, dtype=dtype).reshape(2, 2)
        assert hash_tensor_bytes(tensor) == hash_tensor_bytes(tensor.clone())


def test_hash_tensor_bytes_changes_with_value_dtype_and_shape() -> None:
    tensor = torch.arange(4, dtype=torch.float32)
    changed = tensor.clone()
    changed[0] = 9
    assert hash_tensor_bytes(tensor) != hash_tensor_bytes(changed)
    assert hash_tensor_bytes(tensor) != hash_tensor_bytes(tensor.to(torch.float16))
    assert hash_tensor_bytes(tensor) != hash_tensor_bytes(tensor.reshape(2, 2))
