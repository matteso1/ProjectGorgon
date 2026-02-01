from __future__ import annotations

import math

import torch


def tree_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    dim = q.size(-1)
    scores = (q @ k.T) * (1.0 / math.sqrt(dim))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v