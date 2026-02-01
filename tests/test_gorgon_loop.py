import torch

from gorgon.inference.gorgon_loop import accept_draft_tokens


def test_accept_draft_tokens_greedy_rejects_first_mismatch():
    draft = [1, 2, 3]
    logits = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0],  # argmax -> 1
            [0.0, 0.0, 3.0, 0.0],  # argmax -> 2
            [4.0, 0.0, 0.0, 0.0],  # argmax -> 0 (mismatch)
        ],
        dtype=torch.float32,
    )

    accepted, rejected_at = accept_draft_tokens(draft, logits)

    assert accepted == [1, 2]
    assert rejected_at == 2