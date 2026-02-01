from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parents[3]
_SOURCES = [
    str(_ROOT / "src/gorgon/kernels/cuda/tree_attention.cpp"),
    str(_ROOT / "src/gorgon/kernels/cuda/tree_attention.cu"),
]

_cuda_ext = load(
    name="tree_attention_cuda_ext",
    sources=_SOURCES,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


def tree_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not (q.is_cuda and k.is_cuda and v.is_cuda and mask.is_cuda):
        raise ValueError("q, k, v, mask must be CUDA tensors")

    q = q.contiguous().float()
    k = k.contiguous().float()
    v = v.contiguous().float()
    mask = mask.contiguous().bool()

    return _cuda_ext.forward(q, k, v, mask)[0]