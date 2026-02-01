import math

import torch

from gorgon.kernels.tree_attention_ref import tree_attention_ref
from gorgon.kernels.tree_mask import build_tree_mask

def naive_tree_attention(q, k, v, mask):
    n, d = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(d)
    for row in range(n):
        allowed = mask[row].nonzero(as_tuple=True)[0]
        scores = (q[row] @ k[allowed].T) * scale
        weights = torch.softmax(scores, dim=-1)
        out[row] = weights @ v[allowed]
    return out

def test_tree_attention_ref_matches_naive():
    torch.manual_seed(0)
    parents = [-1, 0, 0]
    mask = build_tree_mask(parents)

    q = torch.randn(3, 4)
    k = torch.randn(3, 4)
    v = torch.randn(3, 4)

    expected = naive_tree_attention(q, k, v, mask)
    actual = tree_attention_ref(q, k, v, mask)

    assert torch.allclose(actual, expected, atol=1e-5)
