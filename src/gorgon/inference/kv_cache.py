from __future__ import annotations

from typing import Iterable, Tuple

import torch


def apply_cache_slice(
    keys: torch.Tensor,
    values: torch.Tensor,
    accepted_indices: Iterable[int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Copy accepted KV entries to the front of the cache.

    Returns:
      new_keys, new_values, new_length
    """
    idx = torch.tensor(list(accepted_indices), device=keys.device, dtype=torch.long)
    new_keys = keys.clone()
    new_values = values.clone()
    new_len = idx.numel()

    if new_len > 0:
        new_keys[:new_len] = keys.index_select(0, idx)
        new_values[:new_len] = values.index_select(0, idx)

    return new_keys, new_values, new_len