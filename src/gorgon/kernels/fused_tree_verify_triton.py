"""Fused tree-attention kernel that computes ancestor masks on-the-fly.

Instead of materializing an (N, N) boolean mask, this kernel walks a
compact ``parents`` int32 array to determine the ancestor relationship
in registers.  This eliminates the mask tensor entirely.

For typical Medusa trees (depth 3-5, N < 100), the ancestor walk is
just 3-5 scalar loads per row, making this much lighter on memory.
"""
from __future__ import annotations

import math
from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_tree_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    parents_ptr,
    out_ptr,
    stride_qm,
    stride_qk,
    stride_km,
    stride_kk,
    stride_vm,
    stride_vk,
    stride_om,
    stride_ok,
    n_cols,
    scale,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
):
    """Fused tree attention: ancestor walk + QKV attention in one kernel.

    For each row i, we walk the parents array to determine which columns j
    are ancestors (or equal to i). Then compute scaled dot-product attention
    with that implicit mask.
    """
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask_cols = offs_n < n_cols

    # Phase 1: Build ancestor bitmask by walking the parents array.
    # is_ancestor[j] = True if j is an ancestor of row (or j == row).
    is_ancestor = (offs_n == row)

    # Walk up from row through parents
    current = tl.load(parents_ptr + row)
    for _ in tl.static_range(0, MAX_DEPTH):
        is_ancestor = is_ancestor | (offs_n == current)
        # Load parent of current (guarded: if current < 0, stay at -1)
        safe_idx = tl.where(current >= 0, current, 0)
        next_parent = tl.load(parents_ptr + safe_idx)
        current = tl.where(current >= 0, next_parent, -1)

    # Phase 2: Q*K scores with ancestor mask, softmax, V accumulation.
    scores = tl.zeros([BLOCK_N], dtype=tl.float32)
    for d_start in tl.static_range(0, DIM, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        q = tl.load(
            q_ptr + row * stride_qm + offs_d * stride_qk,
            mask=offs_d < DIM,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kk,
            mask=mask_cols[:, None] & (offs_d[None, :] < DIM),
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(k * q[None, :], axis=1)

    scores *= scale

    # Apply ancestor mask
    ancestor_mask = is_ancestor & mask_cols
    scores = tl.where(ancestor_mask, scores, -float("inf"))

    # Softmax
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - max_score)
    exp_scores = tl.where(ancestor_mask, exp_scores, 0.0)
    denom = tl.sum(exp_scores, axis=0)
    weights = exp_scores / denom

    # Output = softmax(scores) @ V
    for d_start in tl.static_range(0, DIM, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vk,
            mask=mask_cols[:, None] & (offs_d[None, :] < DIM),
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(v * weights[:, None], axis=0)
        tl.store(
            out_ptr + row * stride_om + offs_d * stride_ok,
            out,
            mask=offs_d < DIM,
        )


def fused_tree_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    parents: List[int] | torch.Tensor,
    max_depth: int = 5,
) -> torch.Tensor:
    """Fused tree attention without materializing the mask tensor.

    Computes tree-masked scaled dot-product attention where the mask
    is derived on-the-fly from the ``parents`` array.

    Args:
        q: (N, D) query tensor, CUDA.
        k: (N, D) key tensor, CUDA.
        v: (N, D) value tensor, CUDA.
        parents: (N,) int32/int64 parent indices (-1 = root).
        max_depth: maximum tree depth (used to unroll ancestor walk).

    Returns:
        (N, D) attention output tensor.
    """
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, v must be CUDA tensors.")

    n_cols, dim = q.shape
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(dim)

    # Convert parents to tensor if needed
    if isinstance(parents, list):
        parents_t = torch.tensor(parents, dtype=torch.int32, device=q.device)
    else:
        parents_t = parents.to(dtype=torch.int32, device=q.device)

    block_n = triton.next_power_of_2(n_cols)
    block_d = 128

    grid = (n_cols,)
    _fused_tree_verify_kernel[grid](
        q,
        k,
        v,
        parents_t,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        n_cols,
        scale,
        DIM=dim,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        MAX_DEPTH=max_depth,
        num_warps=4,
        num_stages=2,
    )
    return out
