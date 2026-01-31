from __future__ import annotations

from typing import List, Tuple


def make_shifted_targets(tokens: List[int], max_heads: int) -> Tuple[List[int], ...]:
    outputs: List[List[int]] = []
    target_length = max(len(tokens) - 1, 0)
    for head_index in range(1, max_heads + 1):
        shifted = tokens[head_index:]
        pad_count = max(target_length - len(shifted), 0)
        outputs.append(shifted + [None] * pad_count)
    return tuple(outputs)
