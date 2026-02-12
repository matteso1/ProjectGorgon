"""Core speculative decoding loop with tree-structured verification.

This module implements the full Medusa-style speculative decoding:
  1. Run backbone on prompt -> get hidden state
  2. Draft candidates via Medusa heads (tree-structured)
  3. Verify all candidates in ONE forward pass using tree attention mask
  4. Accept the longest matching prefix, reject the rest
  5. Repeat until max_new_tokens

Architecture note -- Tree verification
--------------------------------------
The candidate tree is flattened into a 1-D token sequence and appended
to the prompt.  Each candidate's position in this flat sequence has a
known *tree index*.  We build a mapping from tree index -> flat position
so that acceptance checking reads the correct verifier logit for each
candidate.

The verifier logit at flat position ``p`` predicts the token at flat
position ``p + 1``.  For the first draft token the relevant logit is at
the last prompt position (``prompt_len - 1``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import DynamicCache
except ImportError:
    DynamicCache = None

from gorgon.inference.tree_candidates import (
    CandidateTree,
    build_candidate_tree,
    candidate_tree_to_mask,
    get_tree_paths,
)
from gorgon.inference.kv_cache import GorgonKVCache

try:
    from gorgon.kernels.fused_tree_verify_triton import fused_tree_verify as _fused_tree_verify
except ImportError:
    _fused_tree_verify = None


@dataclass
class IterationStats:
    """Per-iteration metrics for a single speculative decoding step."""

    tree_size: int
    accepted_length: int
    head_acceptance: List[bool]
    time_draft_ms: float
    time_verify_ms: float
    time_kv_trim_ms: float


@dataclass
class SpeculativeResult:
    """Result of a full speculative generation run."""

    generated_ids: List[int]
    total_draft_tokens: int
    total_accepted_tokens: int
    num_iterations: int
    iteration_stats: List[IterationStats] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def mean_accepted_length(self) -> float:
        """Mean accepted tokens per iteration (tau)."""
        if not self.iteration_stats:
            return 0.0
        return sum(s.accepted_length for s in self.iteration_stats) / len(
            self.iteration_stats
        )

    @property
    def per_head_acceptance_rates(self) -> List[float]:
        """Acceptance rate for each head position across all iterations."""
        if not self.iteration_stats:
            return []
        max_depth = max(len(s.head_acceptance) for s in self.iteration_stats)
        rates: List[float] = []
        for h in range(max_depth):
            total = 0
            accepted = 0
            for s in self.iteration_stats:
                if h < len(s.head_acceptance):
                    total += 1
                    if s.head_acceptance[h]:
                        accepted += 1
            rates.append(accepted / total if total > 0 else 0.0)
        return rates

    @property
    def tree_utilization(self) -> float:
        """Fraction of tree nodes that were accepted on average."""
        if not self.iteration_stats:
            return 0.0
        utils = []
        for s in self.iteration_stats:
            if s.tree_size > 0:
                utils.append(s.accepted_length / s.tree_size)
        return sum(utils) / len(utils) if utils else 0.0


def accept_draft_tokens(
    draft: List[int],
    logits: torch.Tensor,
) -> Tuple[List[int], int]:
    """Greedy acceptance: accept consecutive draft tokens that match
    the verifier's argmax.  Stop at first mismatch.

    Returns
    -------
    accepted_tokens : list[int]
    rejected_at_index : int
    """
    accepted: List[int] = []
    rejected_at = len(draft)

    for idx, token in enumerate(draft):
        predicted = int(torch.argmax(logits[idx]).item())
        if predicted == token:
            accepted.append(token)
        else:
            rejected_at = idx
            break

    return accepted, rejected_at


# --- Verification --------------------------------------------------------


def _build_flat_position_map(tree: CandidateTree) -> Dict[int, int]:
    """Map each tree-node index to its flat position in the draft chunk.

    Because the candidate tree is stored level-by-level, tree node ``i``
    occupies flat position ``i`` in the appended draft chunk.  The
    ``prompt_len`` offset is applied *outside* this function.
    """
    # Identity mapping -- tree nodes are appended in order.
    return {i: i for i in range(len(tree.parents))}


def _path_to_flat_positions(
    path: List[int],
    pos_map: Dict[int, int],
) -> List[int]:
    """Convert a tree-index path to flat-sequence positions."""
    return [pos_map[ti] for ti in path]


def _tuple_to_dynamic_cache(past_kv_tuple):
    """Convert a tuple-of-tuples KV cache to DynamicCache.

    Modern transformers (>= 4.36) Llama requires DynamicCache objects,
    not raw tuples.  This converts [(K, V), ...] -> DynamicCache.
    """
    if DynamicCache is None:
        return past_kv_tuple  # transformers not installed, pass through
    cache = DynamicCache()
    for layer_idx, layer in enumerate(past_kv_tuple):
        cache.update(layer[0], layer[1], layer_idx)
    return cache


def _trim_kv_cache(past_kv, length: int):
    """Trim KV cache to first `length` positions.

    Always returns a DynamicCache (or None) so that modern transformers
    models can call get_seq_length() on it.
    """
    if past_kv is None:
        return None

    # Modern DynamicCache with crop() (transformers >= 4.46)
    if hasattr(past_kv, 'crop'):
        past_kv.crop(length)
        return past_kv

    # DynamicCache without crop() (transformers 4.36-4.45)
    if hasattr(past_kv, 'key_cache'):
        for layer_idx in range(len(past_kv.key_cache)):
            past_kv.key_cache[layer_idx] = past_kv.key_cache[layer_idx][:, :, :length, :]
            past_kv.value_cache[layer_idx] = past_kv.value_cache[layer_idx][:, :, :length, :]
        if hasattr(past_kv, '_seen_tokens'):
            past_kv._seen_tokens = length
        return past_kv

    # Legacy tuple-of-tuples -> trim then convert to DynamicCache
    trimmed = tuple(
        (layer[0][:, :, :length, :], layer[1][:, :, :length, :])
        for layer in past_kv
    )
    return _tuple_to_dynamic_cache(trimmed)


def _verify_tree_candidates(
    model: nn.Module,
    input_ids: torch.Tensor,
    tree: CandidateTree,
    past_key_values=None,
    use_fused_kernel: bool = False,
) -> Tuple[List[int], int, torch.Tensor, object]:
    """Verify tree-structured candidates with the backbone.

    Runs a single forward pass with all candidate tokens appended
    and uses the verifier logits to find the best accepted path.

    When past_key_values is provided, input_ids should be just the last
    accepted token (the KV cache covers earlier positions). We prepend
    this token to the draft sequence so the model sees [last_token, draft...].

    Returns
    -------
    best_accepted : list[int]
        Token IDs of the longest accepted prefix.
    bonus_token : int
        Verifier's own prediction after the last accepted token.
    next_hidden : Tensor  (1, 1, hidden_dim)
        Hidden state at the position after the last accepted token.
    new_past_kv : past_key_values from the verification forward pass.
    """
    draft_ids = tree.tokens.unsqueeze(0).to(input_ids.device)  # (1, num_candidates)

    if past_key_values is not None:
        # With KV cache: input_ids is just the last token, prepend it to drafts
        last_token = input_ids[:, -1:]
        verifier_input = torch.cat([last_token, draft_ids], dim=1)
        prompt_len = 1  # logit at pos 0 predicts the first draft token
    else:
        # Without KV cache: full context
        verifier_input = torch.cat([input_ids, draft_ids], dim=1)
        prompt_len = input_ids.shape[1]

    with torch.no_grad():
        try:
            outputs = model(
                verifier_input,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values,
            )
        except TypeError:
            # Model doesn't support use_cache/past_key_values kwargs
            # Fall back to full context without cache
            if past_key_values is not None:
                verifier_input = torch.cat([input_ids, draft_ids], dim=1)
                prompt_len = input_ids.shape[1]
            outputs = model(
                verifier_input,
                output_hidden_states=True,
            )

    new_past_kv = getattr(outputs, 'past_key_values', None)
    # Normalize to DynamicCache if model returned a tuple
    if new_past_kv is not None and isinstance(new_past_kv, tuple):
        new_past_kv = _tuple_to_dynamic_cache(new_past_kv)
    verifier_logits = outputs.logits[0]  # (total_seq_len, vocab)

    # Vectorized: compute all predictions once, transfer to CPU once
    all_predictions = torch.argmax(verifier_logits, dim=-1).cpu()
    tree_tokens_cpu = tree.tokens.cpu()

    # Build position map  (tree_idx -> flat offset within draft chunk)
    pos_map = _build_flat_position_map(tree)

    # Get all root-to-leaf paths
    paths = get_tree_paths(tree)

    best_accepted: List[int] = []
    best_bonus_token: int = -1
    winning_path_idx: int = -1

    for path_idx, path in enumerate(paths):
        accepted_tokens: List[int] = []

        for step_idx, tree_idx in enumerate(path):
            parent_tree_idx = tree.parents[tree_idx]
            if parent_tree_idx == -1:
                logit_pos = prompt_len - 1
            else:
                logit_pos = prompt_len + pos_map[parent_tree_idx]

            predicted = int(all_predictions[logit_pos].item())
            draft_token = int(tree_tokens_cpu[tree_idx].item())

            if predicted == draft_token:
                accepted_tokens.append(draft_token)
            else:
                break

        # Bonus: verifier's prediction after the last accepted node
        if accepted_tokens:
            last_accepted_tree_idx = path[len(accepted_tokens) - 1]
            bonus_logit_pos = prompt_len + pos_map[last_accepted_tree_idx]
        else:
            bonus_logit_pos = prompt_len - 1
        bonus = int(all_predictions[bonus_logit_pos].item())

        if len(accepted_tokens) > len(best_accepted):
            best_accepted = accepted_tokens
            best_bonus_token = bonus
            winning_path_idx = path_idx

    # If no tokens accepted on any path, still grab the bonus
    if not best_accepted:
        best_bonus_token = int(all_predictions[prompt_len - 1].item())

    # Hidden state for next iteration
    if best_accepted and winning_path_idx >= 0:
        winning_path = paths[winning_path_idx]
        last_tree_idx = winning_path[len(best_accepted) - 1]
        hidden_pos = prompt_len + pos_map[last_tree_idx]
    else:
        hidden_pos = prompt_len - 1

    next_hidden = outputs.hidden_states[-1][:, hidden_pos : hidden_pos + 1, :]

    return best_accepted, best_bonus_token, next_hidden, new_past_kv


# --- Main generation loop ------------------------------------------------


def speculative_generate(
    model: nn.Module,
    tokenizer,
    heads: nn.ModuleList,
    prompt: str,
    max_new_tokens: int = 128,
    top_k: int = 4,
    prompt_max_length: int = 512,
    device: str = "cuda",
    eos_token_id: int | None = None,
) -> SpeculativeResult:
    """Full speculative generation loop with Medusa heads.

    Args:
        model:       Backbone LLM (e.g. Llama-3-8B in 4-bit).
        tokenizer:   HuggingFace tokenizer.
        heads:       Medusa draft heads (nn.ModuleList).
        prompt:      Input text prompt.
        max_new_tokens: Maximum tokens to generate.
        top_k:       Top-k candidates per head per level.
        prompt_max_length: Max prompt tokens (truncated from left).
        device:      CUDA device string.
        eos_token_id: Token ID for end of sequence.
    """
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > prompt_max_length:
        input_ids = input_ids[:, -prompt_max_length:]
    input_ids = input_ids.to(device)

    generated: List[int] = []
    total_drafted = 0
    total_accepted = 0
    iterations = 0
    all_iteration_stats: List[IterationStats] = []

    # -- Prefill (try with KV cache, fall back without) --------------------
    with torch.no_grad():
        try:
            outputs = model(
                input_ids,
                output_hidden_states=True,
                use_cache=True,
            )
        except TypeError:
            outputs = model(
                input_ids,
                output_hidden_states=True,
            )
    hidden = outputs.hidden_states[-1][:, -1:, :]  # (1, 1, hidden_dim)
    past_key_values = getattr(outputs, 'past_key_values', None)
    # Normalize to DynamicCache if model returned a tuple
    if past_key_values is not None and isinstance(past_key_values, tuple):
        past_key_values = _tuple_to_dynamic_cache(past_key_values)

    # Greedy first token from backbone
    first_token = int(torch.argmax(outputs.logits[0, -1]).item())
    generated.append(first_token)

    if eos_token_id is not None and first_token == eos_token_id:
        return SpeculativeResult(
            generated_ids=generated,
            total_draft_tokens=0,
            total_accepted_tokens=0,
            num_iterations=0,
        )

    # Running context: prompt + generated tokens
    prompt_len = input_ids.shape[1]
    current_ids = torch.cat(
        [
            input_ids,
            torch.tensor(
                [[first_token]], device=device, dtype=input_ids.dtype
            ),
        ],
        dim=1,
    )

    # -- Speculative loop ------------------------------------------------
    while len(generated) < max_new_tokens:
        iterations += 1

        # 1. Draft candidates from Medusa heads
        t_draft_start = time.perf_counter()
        try:
            head_param = next(heads.parameters())
            head_device = head_param.device
            head_dtype = head_param.dtype
        except (AttributeError, StopIteration):
            head_device = hidden.device
            head_dtype = hidden.dtype
        hidden_for_heads = hidden.to(device=head_device, dtype=head_dtype)

        tree = build_candidate_tree(
            heads=heads,
            hidden=hidden_for_heads,
            top_k=top_k,
            max_depth=len(heads),
        )
        total_drafted += len(tree.tokens)
        t_draft_end = time.perf_counter()

        # Handle empty tree (all candidates pruned by adaptive pruning)
        if len(tree.tokens) == 0:
            all_iteration_stats.append(
                IterationStats(
                    tree_size=0,
                    accepted_length=0,
                    head_acceptance=[],
                    time_draft_ms=(t_draft_end - t_draft_start) * 1000.0,
                    time_verify_ms=0.0,
                    time_kv_trim_ms=0.0,
                )
            )
            break

        # 2. Verify all candidates in one forward pass (with KV cache)
        t_verify_start = time.perf_counter()
        accepted, bonus_token, next_hidden, new_past_kv = _verify_tree_candidates(
            model=model,
            input_ids=current_ids,
            tree=tree,
            past_key_values=past_key_values,
        )
        total_accepted += len(accepted)
        t_verify_end = time.perf_counter()

        # Build per-head acceptance: head i accepted if accepted_length > i
        num_heads_used = tree.depth
        head_acceptance = [i < len(accepted) for i in range(num_heads_used)]

        # 3. Collect new tokens (accepted + bonus)
        new_tokens = accepted + [bonus_token]
        tokens_to_append: List[int] = []
        for tok in new_tokens:
            if len(generated) >= max_new_tokens:
                break
            generated.append(tok)
            tokens_to_append.append(tok)
            if eos_token_id is not None and tok == eos_token_id:
                break

        # 4. Stopping conditions
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break
        if len(generated) >= max_new_tokens:
            break

        # 5. Extend context with newly generated tokens
        if tokens_to_append:
            ext = torch.tensor(
                [tokens_to_append], device=device, dtype=input_ids.dtype
            )
            current_ids = torch.cat([current_ids, ext], dim=1)

        # 6. Trim KV cache to accepted positions only
        t_trim_start = time.perf_counter()
        accepted_length = prompt_len + len(generated)
        past_key_values = _trim_kv_cache(new_past_kv, accepted_length)
        t_trim_end = time.perf_counter()

        # 7. Record iteration stats
        all_iteration_stats.append(
            IterationStats(
                tree_size=len(tree.tokens),
                accepted_length=len(accepted),
                head_acceptance=head_acceptance,
                time_draft_ms=(t_draft_end - t_draft_start) * 1000.0,
                time_verify_ms=(t_verify_end - t_verify_start) * 1000.0,
                time_kv_trim_ms=(t_trim_end - t_trim_start) * 1000.0,
            )
        )

        # 8. Reuse hidden state from verification for next draft
        hidden = next_hidden

    return SpeculativeResult(
        generated_ids=generated,
        total_draft_tokens=total_drafted,
        total_accepted_tokens=total_accepted,
        num_iterations=iterations,
        iteration_stats=all_iteration_stats,
    )


# --- Baseline ------------------------------------------------------------


def baseline_generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    prompt_max_length: int = 512,
    device: str = "cuda",
) -> List[int]:
    """Standard autoregressive generation (no speculation) for baseline."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > prompt_max_length:
        input_ids = input_ids[:, -prompt_max_length:]
    input_ids = input_ids.to(device)

    pad_token_id = getattr(tokenizer, "eos_token_id", None)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

    return output_ids[0, input_ids.shape[1] :].tolist()
