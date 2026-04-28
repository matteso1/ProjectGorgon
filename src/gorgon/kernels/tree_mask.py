from __future__ import annotations

from typing import List

import torch


def _is_ancestor(parents: List[int], ancestor: int, node: int) -> bool:
    while node != -1:
        if node == ancestor:
            return True
        node = parents[node]
    return False


def build_tree_mask(parents: List[int], depth: int | None = None) -> torch.Tensor:
    """Build boolean tree attention mask using vectorized ancestor propagation.

    mask[i, j] = True iff node j is an ancestor of node i (or i == j).

    Args:
        parents: parent index for each node (-1 = root).
        depth: max tree depth for iteration bound. If None, uses len(parents)
               as a safe upper bound.
    """
    size = len(parents)
    mask = torch.eye(size, dtype=torch.bool)

    parents_t = torch.tensor(parents, dtype=torch.long)
    max_iter = depth if depth is not None else size

    current = parents_t.clone()
    for _ in range(max_iter):
        valid = current >= 0
        if not valid.any():
            break
        rows = torch.arange(size)[valid]
        cols = current[valid]
        mask[rows, cols] = True
        next_current = torch.full_like(current, -1)
        next_current[valid] = parents_t[current[valid]]
        current = next_current

    return mask
