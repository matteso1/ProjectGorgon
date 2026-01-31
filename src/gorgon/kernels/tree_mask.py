from __future__ import annotations

from typing import List

import torch


def _is_ancestor(parents: List[int], ancestor: int, node: int) -> bool:
    while node != -1:
        if node == ancestor:
            return True
        node = parents[node]
    return False


def build_tree_mask(parents: List[int]) -> torch.Tensor:
    size = len(parents)
    mask = torch.zeros((size, size), dtype=torch.bool)
    for row in range(size):
        for col in range(size):
            if _is_ancestor(parents, col, row):
                mask[row, col] = True
    return mask
