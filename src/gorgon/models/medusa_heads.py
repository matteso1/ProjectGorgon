from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear + SiLU with skip connection.

    Zero-initialized so that at init: linear(x) = 0, SiLU(0) = 0,
    and forward(x) = x + 0 = x (identity).  This means heads start
    by approximating the backbone's own predictions, and training
    only needs to learn the delta for shifted-position prediction.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        # Identity init: zero weight so output starts as identity
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_residual_blocks: int = 1,
        norm: nn.Module | None = None,
    ):
        super().__init__()
        self.norm = norm
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_size) for _ in range(num_residual_blocks)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)
