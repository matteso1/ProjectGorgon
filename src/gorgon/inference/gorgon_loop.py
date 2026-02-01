from __future__ import annotations

from typing import List, Tuple

import torch


def accept_draft_tokens(
    draft: List[int],
    logits: torch.Tensor,
) -> Tuple[List[int], int]:
    """
    Greedy acceptance: accept consecutive draft tokens
    that match the verifier's argmax. Stop at first mismatch.

    Returns:
      accepted_tokens, rejected_at_index
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