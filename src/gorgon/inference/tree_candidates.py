"""Tree-structured candidate generation from Medusa heads.

Given N Medusa heads, each producing top-k token predictions, build a
candidate tree (Cartesian product) and map it into a flat parent array
suitable for tree-attention masking.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class CandidateTree:
    """Flat representation of a speculative candidate tree.

    Attributes:
        tokens:  (num_candidates,) -- token IDs in tree order.
        parents: (num_candidates,) -- parent index for each node (-1 = root).
        depth:   maximum depth of the tree (= number of heads used).
    """
    tokens: torch.Tensor
    parents: List[int]
    depth: int


def _topk_per_head(
    heads: nn.ModuleList,
    hidden: torch.Tensor,
    k: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return top-k token IDs and scores from each head.

    Args:
        heads:  Medusa heads (nn.ModuleList).
        hidden: (1, 1, hidden_dim) -- last hidden state from backbone.
        k:      number of top-k candidates per head.

    Returns:
        list of (ids, scores) tuples, one per head.
        ids: (k,) tensor of token IDs, scores: (k,) tensor of logit values.
    """
    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for head in heads:
        with torch.no_grad():
            logits = head(hidden)  # (1, 1, vocab) or (1, vocab)
            if logits.dim() == 3:
                logits = logits[0, 0]  # (vocab,)
            elif logits.dim() == 2:
                logits = logits[0]     # (vocab,)
            actual_k = min(k, logits.shape[-1])
            scores, ids = torch.topk(logits, actual_k)
            results.append((ids.cpu(), scores.cpu()))
    return results


def build_candidate_tree(
    heads: nn.ModuleList,
    hidden: torch.Tensor,
    top_k: int = 4,
    max_depth: int | None = None,
    max_candidates: int | None = None,
) -> CandidateTree:
    """Build a flat candidate tree from Medusa head predictions.

    The tree structure is:
        root (position 0) -> head-1 candidates -> head-2 candidates -> ...

    For N heads with top-k=4 the tree has 4 + 16 + 64 + ... nodes,
    capped at max_depth levels.

    When max_candidates is set, at each level only the highest-scoring
    parents are expanded until the budget is exhausted.

    Args:
        heads:          Medusa heads.
        hidden:         (1, 1, hidden_dim) last hidden state.
        top_k:          candidates per head per level.
        max_depth:      max tree depth (default: len(heads)).
        max_candidates: total candidate budget (default: None = unlimited).

    Returns:
        CandidateTree with flat tokens, parents, and depth.
    """
    num_heads = len(heads)
    depth = min(max_depth or num_heads, num_heads)

    topk_per_level = _topk_per_head(heads[:depth], hidden, top_k)

    all_tokens: List[int] = []
    all_parents: List[int] = []

    budget_remaining = max_candidates  # None means unlimited

    # Level 1: children of root (index -1 means root).
    level_indices: List[int] = []
    level_scores: List[float] = []
    ids_0, scores_0 = topk_per_level[0]
    for i, tok in enumerate(ids_0.tolist()):
        if budget_remaining is not None and budget_remaining <= 0:
            break
        idx = len(all_tokens)
        all_tokens.append(tok)
        all_parents.append(-1)
        level_indices.append(idx)
        level_scores.append(float(scores_0[i].item()))
        if budget_remaining is not None:
            budget_remaining -= 1

    # Levels 2+: each existing leaf gets top-k children
    for level in range(1, depth):
        if budget_remaining is not None and budget_remaining <= 0:
            break

        ids_l, scores_l = topk_per_level[level]
        tok_list = ids_l.tolist()

        # Sort parents by their score (descending) for budget-aware expansion
        if max_candidates is not None:
            sorted_parents = sorted(
                zip(level_indices, level_scores), key=lambda x: x[1], reverse=True
            )
        else:
            sorted_parents = list(zip(level_indices, level_scores))

        next_level_indices: List[int] = []
        next_level_scores: List[float] = []
        for parent_idx, _ in sorted_parents:
            for i, tok in enumerate(tok_list):
                if budget_remaining is not None and budget_remaining <= 0:
                    break
                idx = len(all_tokens)
                all_tokens.append(tok)
                all_parents.append(parent_idx)
                next_level_indices.append(idx)
                next_level_scores.append(float(scores_l[i].item()))
                if budget_remaining is not None:
                    budget_remaining -= 1
        level_indices = next_level_indices
        level_scores = next_level_scores

    device = hidden.device
    return CandidateTree(
        tokens=torch.tensor(all_tokens, dtype=torch.long, device=device),
        parents=all_parents,
        depth=depth,
    )


def candidate_tree_to_mask(tree: CandidateTree) -> torch.Tensor:
    """Convert a CandidateTree to a boolean attention mask.

    mask[i, j] = True iff node j is an ancestor of node i (or i == j).
    This is passed to the tree-attention kernel.

    Uses vectorized ancestor propagation: O(depth) iterations instead of
    O(n^2) Python loops.
    """
    n = len(tree.parents)
    mask = torch.eye(n, dtype=torch.bool)

    # Build parent tensor for vectorized lookup
    parents = torch.tensor(tree.parents, dtype=torch.long)

    # Walk up the ancestor chain: at each iteration, propagate one more
    # level of ancestry. Max iterations = tree depth.
    current = parents.clone()
    for _ in range(tree.depth):
        valid = current >= 0
        if not valid.any():
            break
        # For each node i where current[i] >= 0, mark that ancestor
        rows = torch.arange(n)[valid]
        cols = current[valid]
        mask[rows, cols] = True
        # Move up: current[i] = parents[current[i]] (if current[i] >= 0)
        next_current = torch.full_like(current, -1)
        next_current[valid] = parents[current[valid]]
        current = next_current

    return mask


def get_tree_paths(tree: CandidateTree) -> List[List[int]]:
    """Extract all root-to-leaf paths from the tree.

    Each path is a list of indices into tree.tokens.
    Used for acceptance checking.
    """
    n = len(tree.parents)
    children: List[List[int]] = [[] for _ in range(n)]
    roots: List[int] = []
    for i, p in enumerate(tree.parents):
        if p == -1:
            roots.append(i)
        else:
            children[p].append(i)

    paths: List[List[int]] = []

    def _dfs(node: int, path: List[int]) -> None:
        path.append(node)
        if not children[node]:
            paths.append(list(path))
        else:
            for child in children[node]:
                _dfs(child, path)
        path.pop()

    for root in roots:
        _dfs(root, [])

    return paths
