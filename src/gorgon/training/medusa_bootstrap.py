from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from gorgon.training.train_heads import replace_none_with_ignore, train_step


def build_head_targets(
    input_ids: torch.Tensor,
    max_heads: int,
    ignore_index: int,
) -> List[torch.Tensor]:
    targets_by_head: List[List[Sequence[int | None]]] = [
        [] for _ in range(max_heads)
    ]
    for row in input_ids.tolist():
        target_length = len(row)
        for head_index in range(1, max_heads + 1):
            shifted = row[head_index:]
            pad_count = max(target_length - len(shifted), 0)
            targets_by_head[head_index - 1].append(shifted + [None] * pad_count)

    return [
        replace_none_with_ignore(rows, ignore_index=ignore_index)
        for rows in targets_by_head
    ]


def train_heads_on_prompts(
    backbone,
    heads,
    tokenizer,
    prompts: Iterable[str],
    steps: int,
    lr: float,
    device: str,
    ignore_index: int = -100,
) -> float:
    optimizer = torch.optim.AdamW(heads.parameters(), lr=lr)
    prompts_list = list(prompts)
    if not prompts_list:
        return 0.0

    heads.to(device)

    loss_value = 0.0
    for step_idx in range(steps):
        prompt = prompts_list[step_idx % len(prompts_list)]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        try:
            backbone_device = next(backbone.parameters()).device
        except StopIteration:
            backbone_device = input_ids.device
        input_ids = input_ids.to(backbone_device)
        target_ids = build_head_targets(input_ids, max_heads=len(heads), ignore_index=ignore_index)
        loss_value = train_step(
            backbone=backbone,
            heads=heads,
            input_ids=input_ids,
            target_ids=target_ids,
            optimizer=optimizer,
            ignore_index=ignore_index,
        )

    return loss_value
