from __future__ import annotations

import time
from typing import Callable, List, Optional

import torch

from gorgon.benchmarks.trials import TrialResult
from gorgon.inference.gorgon_loop import accept_draft_tokens


def truncate_prompt_ids(input_ids: torch.Tensor, prompt_max_length: int) -> torch.Tensor:
    if input_ids.size(1) <= prompt_max_length:
        return input_ids
    return input_ids[:, -prompt_max_length:]


def run_baseline(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    prompt_max_length: int,
    device: str,
    timer: Callable[[], float] = time.perf_counter,
) -> TrialResult:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = truncate_prompt_ids(inputs["input_ids"], prompt_max_length).to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = truncate_prompt_ids(attention_mask, prompt_max_length).to(device)

    pad_token_id = getattr(tokenizer, "eos_token_id", None)

    start = timer()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
    )
    end = timer()

    token_count = int(output_ids.shape[1] - input_ids.shape[1])
    return TrialResult(token_count=token_count, elapsed_s=end - start, acceptance_rate=None)


def _get_head_device(head) -> Optional[torch.device]:
    try:
        parameters = getattr(head, "parameters", None)
        if parameters is None:
            raise AttributeError
        param = next(parameters())
        return param.device
    except (StopIteration, AttributeError, TypeError):
        device_attr = getattr(head, "device", None)
        if device_attr is None:
            return None
        return torch.device(device_attr)


def _get_head_dtype(head) -> Optional[torch.dtype]:
    try:
        parameters = getattr(head, "parameters", None)
        if parameters is None:
            raise AttributeError
        param = next(parameters())
        return param.dtype
    except (StopIteration, AttributeError, TypeError):
        return None


def _ensure_head_device(head, target: torch.device, target_dtype: torch.dtype) -> None:
    head_device = _get_head_device(head)
    if head_device is None or head_device == target:
        try:
            head.to(target, dtype=target_dtype)
        except (TypeError, AttributeError):
            try:
                head.to(target)
            except AttributeError:
                return
        return
    try:
        head.to(target, dtype=target_dtype)
    except (TypeError, AttributeError):
        try:
            head.to(target)
        except AttributeError:
            return


def _extract_draft_tokens(heads, hidden: torch.Tensor, num_medusa_heads: int) -> List[int]:
    draft: List[int] = []
    for head in heads[:num_medusa_heads]:
        head_device = _get_head_device(head)
        head_dtype = _get_head_dtype(head) or hidden.dtype
        if head_device is not None and head_device != hidden.device:
            hidden_for_head = hidden.to(device=head_device, dtype=head_dtype)
        else:
            hidden_for_head = hidden.to(dtype=head_dtype)

        _ensure_head_device(head, hidden_for_head.device, hidden_for_head.dtype)
        logits = head(hidden_for_head)
        token = int(torch.argmax(logits[0]).item())
        draft.append(token)
    return draft


def run_speculative(
    model,
    tokenizer,
    heads,
    prompt: str,
    max_new_tokens: int,
    prompt_max_length: int,
    num_medusa_heads: int,
    device: str,
    timer: Callable[[], float] = time.perf_counter,
) -> TrialResult:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = truncate_prompt_ids(inputs["input_ids"], prompt_max_length).to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = truncate_prompt_ids(attention_mask, prompt_max_length).to(device)

    start = timer()
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden = outputs.hidden_states[-1][:, -1:, :]
    draft = _extract_draft_tokens(heads, hidden, num_medusa_heads)

    if not draft:
        end = timer()
        return TrialResult(token_count=0, elapsed_s=end - start, acceptance_rate=0.0)

    draft_ids = torch.tensor([draft], device=input_ids.device, dtype=input_ids.dtype)
    verifier_input = torch.cat([input_ids, draft_ids], dim=1)
    if attention_mask is not None:
        draft_mask = torch.ones_like(draft_ids, device=attention_mask.device)
        verifier_mask = torch.cat([attention_mask, draft_mask], dim=1)
    else:
        verifier_mask = None

    verifier_outputs = model(
        verifier_input,
        attention_mask=verifier_mask,
        output_hidden_states=False,
    )
    start_idx = input_ids.shape[1] - 1
    end_idx = start_idx + len(draft)
    verifier_logits = verifier_outputs.logits[0, start_idx:end_idx, :]
    accepted, _ = accept_draft_tokens(draft, verifier_logits)
    end = timer()

    token_count = min(max_new_tokens, len(accepted))
    acceptance_rate: Optional[float] = None
    if draft:
        acceptance_rate = len(accepted) / len(draft)

    return TrialResult(
        token_count=token_count,
        elapsed_s=end - start,
        acceptance_rate=acceptance_rate,
    )
