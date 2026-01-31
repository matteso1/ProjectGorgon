from __future__ import annotations

import torch
import torch.nn as nn


class MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, vocab_size),
        )
        for layer in self.block:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
