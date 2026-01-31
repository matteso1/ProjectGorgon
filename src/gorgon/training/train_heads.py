from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


def replace_none_with_ignore(
    targets: Sequence[Sequence[int | None]],
    ignore_index: int = -100,
) -> torch.Tensor:
    return torch.tensor(
        [[value if value is not None else ignore_index for value in row] for row in targets],
        dtype=torch.long,
    )


def compute_heads_loss(
    logits_list: Sequence[torch.Tensor],
    target_ids: Sequence[torch.Tensor],
    ignore_index: int = -100,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    losses: List[torch.Tensor] = []
    for logits, targets in zip(logits_list, target_ids):
        vocab_size = logits.size(-1)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        losses.append(loss)
    return sum(losses)


def train_step(
    backbone: nn.Module,
    heads: nn.ModuleList,
    input_ids: torch.Tensor,
    target_ids: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    ignore_index: int = -100,
) -> float:
    backbone.eval()
    heads.train()
    with torch.no_grad():
        outputs = backbone(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]

    logits_list = [head(hidden_states) for head in heads]
    loss = compute_heads_loss(logits_list, target_ids, ignore_index=ignore_index)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach().item()
