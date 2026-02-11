from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
from tqdm import tqdm


def _head_device_and_dtype(heads: nn.ModuleList, fallback: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    for param in heads.parameters():
        return param.device, param.dtype
    return fallback.device, fallback.dtype


def distill_heads_last_token(
    backbone: nn.Module,
    heads: nn.ModuleList,
    tokenizer,
    prompts: Iterable[str],
    steps: int,
    lr: float,
    device: str,
) -> float:
    optimizer = torch.optim.AdamW(heads.parameters(), lr=lr)
    prompts_list = list(prompts)
    if not prompts_list:
        return 0.0

    heads.to(device)

    loss_value = 0.0
    for step_idx in tqdm(range(steps), desc="distill", unit="step"):
        prompt = prompts_list[step_idx % len(prompts_list)]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        try:
            backbone_device = next(backbone.parameters()).device
        except StopIteration:
            backbone_device = input_ids.device
        input_ids = input_ids.to(backbone_device)

        with torch.no_grad():
            try:
                outputs = backbone(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            except TypeError:
                outputs = backbone(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

        hidden_last = outputs.hidden_states[-1][:, -1, :]
        target_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        head_device, head_dtype = _head_device_and_dtype(heads, hidden_last)
        hidden_last = hidden_last.to(device=head_device, dtype=head_dtype)
        target_ids = target_ids.to(head_device)

        logits_list: List[torch.Tensor] = [head(hidden_last) for head in heads]
        loss = sum(
            nn.functional.cross_entropy(logits, target_ids) for logits in logits_list
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())

    return loss_value
