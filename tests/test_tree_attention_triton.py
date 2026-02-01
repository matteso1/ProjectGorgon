import torch

from gorgon.kernels.tree_attention_ref import tree_attention_ref
from gorgon.kernels.tree_attention_triton import tree_attention_triton
from gorgon.kernels.tree_mask import build_tree_mask


def test_tree_attention_triton_matches_ref():
    torch.manual_seed(0)
    parents = [-1, 0, 0]
    mask = build_tree_mask(parents).to("cuda")

    q = torch.randn(3, 4, device="cuda")
    k = torch.randn(3, 4, device="cuda")
    v = torch.randn(3, 4, device="cuda")

    expected = tree_attention_ref(q, k, v, mask)
    actual = tree_attention_triton(q, k, v, mask)

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)