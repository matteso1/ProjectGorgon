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
        tokens:  (num_candidates,) — token IDs in tree order.
        parents: (num_candidates,) — parent index for each node (-1 = root).
        depth:   maximum depth of the tree (= number of heads used).
    """
    tokens: torch.Tensor
    parents: List[int]
    depth: int


def _topk_per_head(
    heads: nn.ModuleList,
    hidden: torch.Tensor,
    k: int,
) -> List[torch.Tensor]:
    """Return top-k token IDs from each head.

    Args:
        heads:  Medusa heads (nn.ModuleList).
        hidden: (1, 1, hidden_dim) — last hidden state from backbone.
        k:      number of top-k candidates per head.

    Returns:
        list of (k,) tensors, one per head, containing token IDs.
    """
    topk_ids: List[torch.Tensor] = []
    for head in heads:
        with torch.no_grad():
            logits = head(hidden)  # (1, 1, vocab) or (1, vocab)
            if logits.dim() == 3:
                logits = logits[0, 0]  # (vocab,)
            elif logits.dim() == 2:
                logits = logits[0]     # (vocab,)
            _, ids = torch.topk(logits, k)
            topk_ids.append(ids.cpu())
    return topk_ids


def build_candidate_tree(
    heads: nn.ModuleList,
    hidden: torch.Tensor,
    top_k: int = 4,
    max_depth: int | None = None,
) -> CandidateTree:
    """Build a flat candidate tree from Medusa head predictions.

    The tree structure is:
        root (position 0) → head-1 candidates → head-2 candidates → ...

    For N heads with top-k=4 the tree has 1 + 4 + 16 + 64 + ... nodes,
    capped at max_depth levels.

    Each path from root to leaf represents a possible continuation
    of the sequence (one token per speculative step).

    Args:
        heads:     Medusa heads.
        hidden:    (1, 1, hidden_dim) last hidden state.
        top_k:     candidates per head per level.
        max_depth: max tree depth (default: len(heads)).

    Returns:
        CandidateTree with flat tokens, parents, and depth.
    """
    num_heads = len(heads)
    depth = min(max_depth or num_heads, num_heads)

    topk_per_level = _topk_per_head(heads[:depth], hidden, top_k)

    # Build the tree level by level.
    # Level 0: root (placeholder token, not actually generated).
    # Level d: Cartesian product of top-k choices from head d,
    #          attached to each leaf at level d-1.
    all_tokens: List[int] = []
    all_parents: List[int] = []

    # Level 1: children of root (index -1 means root).
    level_indices: List[int] = []
    for tok in topk_per_level[0].tolist():
        idx = len(all_tokens)
        all_tokens.append(tok)
        all_parents.append(-1)  # parent is root (not in list)
        level_indices.append(idx)

    # Levels 2+: each existing leaf gets top-k children
    for level in range(1, depth):
        next_level_indices: List[int] = []
        for parent_idx in level_indices:
            for tok in topk_per_level[level].tolist():
                idx = len(all_tokens)
                all_tokens.append(tok)
                all_parents.append(parent_idx)
                next_level_indices.append(idx)
        level_indices = next_level_indices

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
    """
    n = len(tree.parents)
    mask = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        # Self
        mask[i, i] = True
        # Walk up the ancestor chain
        node = tree.parents[i]
        while node != -1:
            mask[i, node] = True
            node = tree.parents[node]
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
