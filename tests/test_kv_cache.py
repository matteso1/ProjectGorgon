import torch

from gorgon.inference.kv_cache import apply_cache_slice


def test_apply_cache_slice_copies_branch_to_front():
    keys = torch.tensor([10, 11, 12, 13, 14], dtype=torch.float32)
    values = torch.tensor([110, 111, 112, 113, 114], dtype=torch.float32)

    new_k, new_v, new_len = apply_cache_slice(keys, values, [0, 2, 4])

    assert new_len == 3
    assert torch.equal(new_k[:3], torch.tensor([10, 12, 14], dtype=torch.float32))
    assert torch.equal(new_v[:3], torch.tensor([110, 112, 114], dtype=torch.float32))