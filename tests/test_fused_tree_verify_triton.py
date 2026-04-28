"""Tests for the fused tree verification Triton kernel.

Compares against the reference tree_attention_ref + build_tree_mask path
to verify numerical correctness.
"""
from __future__ import annotations

import pytest
import torch

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

HAS_CUDA = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not HAS_CUDA or not HAS_TRITON,
    reason="CUDA and Triton required for fused kernel tests",
)

from gorgon.kernels.tree_attention_ref import tree_attention_ref
from gorgon.kernels.tree_mask import build_tree_mask
from gorgon.kernels.fused_tree_verify_triton import fused_tree_verify


def _make_tree_data(parents, dim, dtype=torch.float32):
    """Create random Q, K, V tensors for a tree of given shape."""
    n = len(parents)
    q = torch.randn(n, dim, dtype=dtype, device="cuda")
    k = torch.randn(n, dim, dtype=dtype, device="cuda")
    v = torch.randn(n, dim, dtype=dtype, device="cuda")
    return q, k, v


# -- Basic correctness tests --


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_fused_matches_ref_simple_chain(dim):
    """Linear chain: 0 -> 1 -> 2 -> 3."""
    parents = [-1, 0, 1, 2]
    q, k, v = _make_tree_data(parents, dim)

    mask = build_tree_mask(parents).to("cuda")
    ref_out = tree_attention_ref(q, k, v, mask)
    fused_out = fused_tree_verify(q, k, v, parents, max_depth=4)

    torch.testing.assert_close(fused_out, ref_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_fused_matches_ref_binary_tree(dim):
    """Binary tree: root=0, children of 0 are 1,2; children of 1 are 3,4."""
    parents = [-1, 0, 0, 1, 1]
    q, k, v = _make_tree_data(parents, dim)

    mask = build_tree_mask(parents).to("cuda")
    ref_out = tree_attention_ref(q, k, v, mask)
    fused_out = fused_tree_verify(q, k, v, parents, max_depth=3)

    torch.testing.assert_close(fused_out, ref_out, atol=1e-4, rtol=1e-4)


def test_fused_matches_ref_wide_tree():
    """Wide tree: 4 roots, each with 3 children = 16 nodes."""
    parents = [-1, -1, -1, -1,  # 4 roots
               0, 0, 0,          # children of 0
               1, 1, 1,          # children of 1
               2, 2, 2,          # children of 2
               3, 3, 3]          # children of 3
    q, k, v = _make_tree_data(parents, 64)

    mask = build_tree_mask(parents).to("cuda")
    ref_out = tree_attention_ref(q, k, v, mask)
    fused_out = fused_tree_verify(q, k, v, parents, max_depth=2)

    torch.testing.assert_close(fused_out, ref_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n,dim", [
    (1, 32),
    (7, 64),
    (20, 128),
    (50, 64),
])
def test_fused_matches_ref_medusa_like(n, dim):
    """Random Medusa-like tree: each node except root has a parent < itself."""
    import random
    random.seed(42)
    parents = [-1]
    for i in range(1, n):
        parents.append(random.randint(0, i - 1))

    q, k, v = _make_tree_data(parents, dim)

    mask = build_tree_mask(parents).to("cuda")
    ref_out = tree_attention_ref(q, k, v, mask)
    fused_out = fused_tree_verify(q, k, v, parents, max_depth=n)

    torch.testing.assert_close(fused_out, ref_out, atol=1e-4, rtol=1e-4)


def test_fused_accepts_tensor_parents():
    """Verify the kernel accepts parents as a torch.Tensor."""
    parents_list = [-1, 0, 0, 1, 1]
    parents_tensor = torch.tensor(parents_list, dtype=torch.int32, device="cuda")

    q, k, v = _make_tree_data(parents_list, 64)

    out_list = fused_tree_verify(q, k, v, parents_list, max_depth=3)
    out_tensor = fused_tree_verify(q, k, v, parents_tensor, max_depth=3)

    torch.testing.assert_close(out_list, out_tensor, atol=1e-6, rtol=1e-6)


def test_fused_single_node():
    """Single node tree (just the root)."""
    parents = [-1]
    q, k, v = _make_tree_data(parents, 64)

    mask = build_tree_mask(parents).to("cuda")
    ref_out = tree_attention_ref(q, k, v, mask)
    fused_out = fused_tree_verify(q, k, v, parents, max_depth=1)

    torch.testing.assert_close(fused_out, ref_out, atol=1e-4, rtol=1e-4)
