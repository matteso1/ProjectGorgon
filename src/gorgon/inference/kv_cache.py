"""KV-cache management for speculative decoding.

Provides a GorgonKVCache wrapper that:
  - Stores HuggingFace-style past_key_values tuples
  - Handles rollback on rejected speculative branches
  - Keeps only accepted token positions after verification
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


# HuggingFace past_key_values:
# tuple of (num_layers,) tuples of (key, value) tensors
# each tensor shape: (batch, num_heads, seq_len, head_dim)
PastKV = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class GorgonKVCache:
    """Manages KV-cache state across speculative decoding iterations.

    The cache wraps HuggingFace-style `past_key_values` and provides
    operations for:
      - Saving checkpoints before speculative steps
      - Rolling back to the checkpoint on rejection
      - Trimming to only keep accepted positions
    """

    def __init__(self) -> None:
        self._cache: Optional[PastKV] = None
        self._checkpoint: Optional[PastKV] = None
        self._seq_len: int = 0

    @property
    def past_key_values(self) -> Optional[PastKV]:
        return self._cache

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(self, past_kv: PastKV) -> None:
        """Store new past_key_values from a model forward pass."""
        self._cache = past_kv
        if past_kv and past_kv[0][0].numel() > 0:
            self._seq_len = past_kv[0][0].shape[2]
        else:
            self._seq_len = 0

    def checkpoint(self) -> None:
        """Save current cache state so we can rollback after rejection."""
        if self._cache is None:
            self._checkpoint = None
            return
        # Deep copy the tensors (they're not huge for a single seq)
        self._checkpoint = tuple(
            (k.clone(), v.clone()) for k, v in self._cache
        )

    def rollback(self) -> None:
        """Restore cache to last checkpoint."""
        if self._checkpoint is not None:
            self._cache = self._checkpoint
            self._checkpoint = None
            if self._cache and self._cache[0][0].numel() > 0:
                self._seq_len = self._cache[0][0].shape[2]
            else:
                self._seq_len = 0

    def trim_to(self, length: int) -> None:
        """Keep only the first `length` positions in the cache."""
        if self._cache is None:
            return
        self._cache = tuple(
            (k[:, :, :length, :], v[:, :, :length, :])
            for k, v in self._cache
        )
        self._seq_len = length

    def clear(self) -> None:
        """Reset the cache entirely."""
        self._cache = None
        self._checkpoint = None
        self._seq_len = 0


def apply_cache_slice(
    keys: torch.Tensor,
    values: torch.Tensor,
    accepted_indices: list[int],
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Copy accepted KV entries to the front of the cache.

    Returns:
      new_keys, new_values, new_length
    """
    idx = torch.tensor(accepted_indices, device=keys.device, dtype=torch.long)
    new_keys = keys.clone()
    new_values = values.clone()
    new_len = idx.numel()

    if new_len > 0:
        new_keys[:new_len] = keys.index_select(0, idx)
        new_values[:new_len] = values.index_select(0, idx)

    return new_keys, new_values, new_len