"""Benchmark inference routines.

Provides `run_baseline()` and `run_speculative()` that return
`TrialResult` objects for the benchmark harness.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

import torch

from gorgon.benchmarks.trials import TrialResult
from gorgon.inference.gorgon_loop import (
    accept_draft_tokens,
    speculative_generate,
    SpeculativeResult,
)


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
    top_k: int = 4,
) -> TrialResult:
    """Run speculative decoding using the full Gorgon loop.

    Uses the tree-structured speculative generation loop for proper
    multi-head drafting and verification.
    """
    start = timer()

    result: SpeculativeResult = speculative_generate(
        model=model,
        tokenizer=tokenizer,
        heads=heads,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        prompt_max_length=prompt_max_length,
        device=device,
    )

    end = timer()

    # Extract enhanced metrics from iteration stats
    mean_accepted_length = None
    per_head_acceptance = None
    tree_utilization = None
    time_breakdown = None

    if result.iteration_stats:
        mean_accepted_length = result.mean_accepted_length
        per_head_acceptance = result.per_head_acceptance_rates
        tree_utilization = result.tree_utilization

        total_draft_ms = sum(s.time_draft_ms for s in result.iteration_stats)
        total_verify_ms = sum(s.time_verify_ms for s in result.iteration_stats)
        total_trim_ms = sum(s.time_kv_trim_ms for s in result.iteration_stats)
        time_breakdown = {
            "draft_ms": total_draft_ms,
            "verify_ms": total_verify_ms,
            "kv_trim_ms": total_trim_ms,
        }

    return TrialResult(
        token_count=len(result.generated_ids),
        elapsed_s=end - start,
        acceptance_rate=result.acceptance_rate,
        mean_accepted_length=mean_accepted_length,
        per_head_acceptance=per_head_acceptance,
        tree_utilization=tree_utilization,
        time_breakdown=time_breakdown,
    )
