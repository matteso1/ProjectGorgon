"""Data utilities for Medusa head training.

Provides a streaming dataset wrapper that tokenizes text into
fixed-length chunks suitable for parallel head training.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset


def make_shifted_targets(tokens: List[int], max_heads: int) -> Tuple[List[int], ...]:
    """Create shifted target sequences for each Medusa head.

    Head k predicts token at position (t + k), so target for head k
    is the input shifted forward by k positions.
    """
    outputs: List[List[int]] = []
    target_length = max(len(tokens) - 1, 0)
    for head_index in range(1, max_heads + 1):
        shifted = tokens[head_index:]
        pad_count = max(target_length - len(shifted), 0)
        outputs.append(shifted + [None] * pad_count)
    return tuple(outputs)


class GorgonDataset(IterableDataset):
    """Streaming dataset for Medusa head training.

    Wraps a HuggingFace dataset and yields tokenized chunks of fixed
    length. Handles both streaming and non-streaming modes, and
    auto-detects the text field name.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        seq_length: int = 512,
        split: str = "train",
        streaming: bool = True,
        max_samples: Optional[int] = None,
        text_field: str = "text",
        subset: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split
        self.streaming = streaming
        self.max_samples = max_samples
        self.text_field = text_field
        self.subset = subset

    def _load_dataset(self):
        from datasets import load_dataset

        kwargs = {
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.subset:
            kwargs["name"] = self.subset

        try:
            ds = load_dataset(self.dataset_name, **kwargs)
        except Exception:
            try:
                # Some datasets don't support streaming; fall back
                kwargs["streaming"] = False
                self.streaming = False
                ds = load_dataset(self.dataset_name, **kwargs)
            except (ValueError, KeyError):
                # Dataset may use non-standard split names (e.g. train_sft);
                # load without split and pick the first available.
                kwargs.pop("split", None)
                kwargs["streaming"] = False
                self.streaming = False
                ds = load_dataset(self.dataset_name, **kwargs)
                if hasattr(ds, "keys"):
                    # Prefer train-like splits
                    splits = list(ds.keys())
                    for candidate in ("train", "train_sft", "train_gen"):
                        if candidate in splits:
                            ds = ds[candidate]
                            break
                    else:
                        ds = ds[splits[0]]

        return ds

    def _get_text(self, sample: dict) -> str:
        """Extract text from sample, auto-detecting the field name.

        Handles both plain-text datasets (WikiText, C4) and conversation
        datasets (ShareGPT) where each sample has a ``conversations``
        list of ``{"from": ..., "value": ...}`` turns.
        """
        # Conversation-format datasets (ShareGPT, UltraChat, etc.)
        for conv_field in ("conversations", "messages"):
            if conv_field in sample and isinstance(sample[conv_field], list):
                parts: list[str] = []
                for turn in sample[conv_field]:
                    if not isinstance(turn, dict):
                        continue
                    # Handle both ShareGPT (from/value) and UltraChat (role/content)
                    value = turn.get("value") or turn.get("content") or ""
                    if value:
                        parts.append(value)
                if parts:
                    return "\n\n".join(parts)

        if self.text_field in sample:
            return sample[self.text_field] or ""
        # Try common field names
        for field in ("text", "content", "document", "sentence"):
            if field in sample and sample[field]:
                return sample[field]
        return ""

    def __iter__(self) -> Iterator[torch.Tensor]:
        ds = self._load_dataset()
        buffer: List[int] = []
        count = 0

        for sample in ds:
            if self.max_samples is not None and count >= self.max_samples:
                break

            text = self._get_text(sample)
            if not text or len(text) < 50:  # Skip very short texts
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_length:
                chunk = buffer[:self.seq_length]
                buffer = buffer[self.seq_length:]
                yield torch.tensor(chunk, dtype=torch.long)
                count += 1

                if self.max_samples is not None and count >= self.max_samples:
                    return


# Mapping of friendly names → (HuggingFace dataset path, subset, text_field)
# Verified working as of Feb 2026
DATASET_REGISTRY = {
    "wikitext": ("wikitext", "wikitext-103-raw-v1", "text"),
    "sharegpt": ("HuggingFaceH4/ultrachat_200k", None, "messages"),
    "openwebtext": ("Skylion007/openwebtext", None, "text"),
    "c4": ("allenai/c4", "en", "text"),
    "redpajama": ("togethercomputer/RedPajama-Data-V2", "sample-10B", "raw_content"),
    "slim_pajama": ("cerebras/SlimPajama-627B", None, "text"),
}


def get_dataloader(
    name: str,
    tokenizer,
    seq_length: int = 512,
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    max_samples: Optional[int] = None,
) -> "torch.utils.data.DataLoader":
    """Create a DataLoader wrapping a GorgonDataset."""
    from torch.utils.data import DataLoader

    dataset = get_dataset(
        name=name,
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_dataset(
    name: str,
    tokenizer,
    seq_length: int = 512,
    max_samples: Optional[int] = None,
) -> GorgonDataset:
    """Get a training dataset by friendly name.

    Falls back to treating `name` as a HuggingFace dataset path if
    not found in the registry.
    """
    if name in DATASET_REGISTRY:
        dataset_path, subset, text_field = DATASET_REGISTRY[name]
    else:
        dataset_path = name
        subset = None
        text_field = "text"

    return GorgonDataset(
        dataset_name=dataset_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        max_samples=max_samples,
        text_field=text_field,
        subset=subset,
    )
