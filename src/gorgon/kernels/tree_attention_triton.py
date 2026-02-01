from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _tree_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    stride_qm,
    stride_qk,
    stride_km,
    stride_kk,
    stride_vm,
    stride_vk,
    stride_mm,
    stride_mn,
    stride_om,
    stride_ok,
    n_cols,
    scale,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask_cols = offs_n < n_cols

    # compute attention scores: q·k
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

    mask_vals = tl.load(
        mask_ptr + row * stride_mm + offs_n * stride_mn,
        mask=mask_cols,
        other=0,
    )
    mask_vals = mask_vals.to(tl.int1)
    scores = tl.where(mask_vals, scores, -float("inf"))

    # softmax
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - max_score)
    exp_scores = tl.where(mask_vals, exp_scores, 0.0)
    denom = tl.sum(exp_scores, axis=0)
    weights = exp_scores / denom

    # compute output = softmax(scores) @ V
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


def tree_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not mask.is_cuda:
        raise ValueError("q, k, v, and mask must be CUDA tensors.")

    n_cols, dim = q.shape
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(dim)

    block_n = triton.next_power_of_2(n_cols)
    block_d = 128

    grid = (n_cols,)
    _tree_attention_kernel[grid](
        q,
        k,
        v,
        mask,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        mask.stride(0),
        mask.stride(1),
        out.stride(0),
        out.stride(1),
        n_cols,
        scale,
        DIM=dim,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out