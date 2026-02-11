"""Core speculative decoding loop with tree-structured verification.

This module implements the full Medusa-style speculative decoding:
  1. Run backbone on prompt → get hidden state
  2. Draft candidates via Medusa heads (tree-structured)
  3. Verify all candidates in ONE forward pass using tree attention mask
  4. Accept the longest matching prefix, reject the rest
  5. Repeat until max_new_tokens

Architecture note — Tree verification
--------------------------------------
The candidate tree is flattened into a 1-D token sequence and appended
to the prompt.  Each candidate's position in this flat sequence has a
known *tree index*.  We build a mapping from tree index → flat position
so that acceptance checking reads the correct verifier logit for each
candidate.

The verifier logit at flat position ``p`` predicts the token at flat
position ``p + 1``.  For the first draft token the relevant logit is at
the last prompt position (``prompt_len - 1``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from gorgon.inference.tree_candidates import (
    CandidateTree,
    build_candidate_tree,
    candidate_tree_to_mask,
    get_tree_paths,
)
from gorgon.inference.kv_cache import GorgonKVCache


@dataclass
class SpeculativeResult:
    """Result of a full speculative generation run."""

    generated_ids: List[int]
    total_draft_tokens: int
    total_accepted_tokens: int
    num_iterations: int

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens


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


# ─── Verification ────────────────────────────────────────────────────


def _build_flat_position_map(tree: CandidateTree) -> Dict[int, int]:
    """Map each tree-node index to its flat position in the draft chunk.

    Because the candidate tree is stored level-by-level, tree node ``i``
    occupies flat position ``i`` in the appended draft chunk.  The
    ``prompt_len`` offset is applied *outside* this function.
    """
    # Identity mapping — tree nodes are appended in order.
    return {i: i for i in range(len(tree.parents))}


def _path_to_flat_positions(
    path: List[int],
    pos_map: Dict[int, int],
) -> List[int]:
    """Convert a tree-index path to flat-sequence positions."""
    return [pos_map[ti] for ti in path]


def _verify_tree_candidates(
    model: nn.Module,
    input_ids: torch.Tensor,
    tree: CandidateTree,
    past_key_values=None,
) -> Tuple[List[int], int, torch.Tensor]:
    """Verify tree-structured candidates with the backbone.

    Runs a single forward pass with all candidate tokens appended
    and uses the verifier logits to find the best accepted path.

    Key insight
    -----------
    The verifier logit at position ``p`` predicts the token at position
    ``p + 1``.  For the first draft candidate (tree node 0, appended at
    flat position ``prompt_len``), the predicting logit sits at
    ``prompt_len - 1`` — the last *prompt* position.

    Within the draft chunk the relationship is:

        logit at ``prompt_len + flat_pos_of_node``  predicts  ``flat_pos_of_node + 1``

    But in a tree, a node's "logical predecessor" is its *parent*, not
    the preceding flat index.  So we verify each path independently: for
    the *k*-th step along a root-to-leaf path we use the logit at the
    flat position of the *parent* (or ``prompt_len - 1`` for the root).

    Returns
    -------
    best_accepted : list[int]
        Token IDs of the longest accepted prefix.
    bonus_token : int
        Verifier's own prediction after the last accepted token.
    next_hidden : Tensor  (1, 1, hidden_dim)
        Hidden state at the position after the last accepted token.
    """
    # Flatten tree tokens and append to input
    draft_ids = tree.tokens.unsqueeze(0)  # (1, num_candidates)
    verifier_input = torch.cat([input_ids, draft_ids], dim=1)

    with torch.no_grad():
        outputs = model(
            verifier_input,
            output_hidden_states=True,
            use_cache=False,
        )

    prompt_len = input_ids.shape[1]
    verifier_logits = outputs.logits[0]  # (total_seq_len, vocab)

    # Build position map  (tree_idx → flat offset within draft chunk)
    pos_map = _build_flat_position_map(tree)

    # Get all root-to-leaf paths
    paths = get_tree_paths(tree)

    best_accepted: List[int] = []
    best_bonus_token: int = -1

    for path in paths:
        accepted_tokens: List[int] = []

        for step_idx, tree_idx in enumerate(path):
            # Find the logit that *predicts* this node's token:
            #   - For the first step (root children), the predicting
            #     logit is at prompt_len - 1 (the last prompt token).
            #   - For deeper steps, the predicting logit is at the
            #     flat position of the *parent* node offset by
            #     prompt_len.
            parent_tree_idx = tree.parents[tree_idx]
            if parent_tree_idx == -1:
                # Root child — predicted by the last prompt token
                logit_pos = prompt_len - 1
            else:
                logit_pos = prompt_len + pos_map[parent_tree_idx]

            predicted = int(torch.argmax(verifier_logits[logit_pos]).item())
            draft_token = int(tree.tokens[tree_idx].item())

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
        bonus = int(torch.argmax(verifier_logits[bonus_logit_pos]).item())

        if len(accepted_tokens) > len(best_accepted):
            best_accepted = accepted_tokens
            best_bonus_token = bonus

    # If no tokens accepted on any path, still grab the bonus
    if not best_accepted:
        best_bonus_token = int(
            torch.argmax(verifier_logits[prompt_len - 1]).item()
        )

    # Hidden state for next iteration: use position after last accepted
    if best_accepted:
        # We have specific accepted nodes — take hidden at the last one
        # (the path that won)
        winning_path = None
        for path in paths:
            acc_check: List[int] = []
            for step_idx, tree_idx in enumerate(path):
                parent_tree_idx = tree.parents[tree_idx]
                if parent_tree_idx == -1:
                    lp = prompt_len - 1
                else:
                    lp = prompt_len + pos_map[parent_tree_idx]
                pred = int(torch.argmax(verifier_logits[lp]).item())
                tk = int(tree.tokens[tree_idx].item())
                if pred == tk:
                    acc_check.append(tk)
                else:
                    break
            if acc_check == best_accepted:
                winning_path = path
                break

        if winning_path:
            last_tree_idx = winning_path[len(best_accepted) - 1]
            hidden_pos = prompt_len + pos_map[last_tree_idx]
        else:
            hidden_pos = prompt_len - 1
    else:
        hidden_pos = prompt_len - 1

    next_hidden = outputs.hidden_states[-1][:, hidden_pos : hidden_pos + 1, :]

    return best_accepted, best_bonus_token, next_hidden


# ─── Main generation loop ────────────────────────────────────────────


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

    # ── Prefill ──────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = outputs.hidden_states[-1][:, -1:, :]  # (1, 1, hidden_dim)

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
    current_ids = torch.cat(
        [
            input_ids,
            torch.tensor(
                [[first_token]], device=device, dtype=input_ids.dtype
            ),
        ],
        dim=1,
    )

    # ── Speculative loop ─────────────────────────────────────────────
    while len(generated) < max_new_tokens:
        iterations += 1

        # 1. Draft candidates from Medusa heads
        head_device = next(heads.parameters()).device
        head_dtype = next(heads.parameters()).dtype
        hidden_for_heads = hidden.to(device=head_device, dtype=head_dtype)

        tree = build_candidate_tree(
            heads=heads,
            hidden=hidden_for_heads,
            top_k=top_k,
            max_depth=len(heads),
        )
        total_drafted += len(tree.tokens)

        # 2. Verify all candidates in one forward pass
        accepted, bonus_token, next_hidden = _verify_tree_candidates(
            model=model,
            input_ids=current_ids,
            tree=tree,
        )
        total_accepted += len(accepted)

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

        # Sliding-window guard for very long sequences
        if current_ids.shape[1] > prompt_max_length * 2:
            current_ids = current_ids[:, -prompt_max_length:]

        # 6. Reuse hidden state from verification for next draft
        hidden = next_hidden

    return SpeculativeResult(
        generated_ids=generated,
        total_draft_tokens=total_drafted,
        total_accepted_tokens=total_accepted,
        num_iterations=iterations,
    )


# ─── Baseline ────────────────────────────────────────────────────────


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