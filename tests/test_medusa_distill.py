from types import SimpleNamespace

import torch
import torch.nn as nn

from gorgon.models.medusa_heads import MedusaHead
from gorgon.training.medusa_distill import distill_heads_last_token, distill_heads


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, input_ids, output_hidden_states=True, return_dict=True, **_):
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.hidden_size)
        logits = torch.zeros(batch, seq, self.vocab_size)
        logits[:, -1, 1] = 1.0
        return SimpleNamespace(hidden_states=[hidden], logits=logits)


class DummyTokenizer:
    def __call__(self, text: str, return_tensors: str):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


def test_distill_heads_last_token_runs() -> None:
    backbone = DummyBackbone(hidden_size=4, vocab_size=8)
    heads = nn.ModuleList([MedusaHead(hidden_size=4, vocab_size=8)])
    tokenizer = DummyTokenizer()

    loss = distill_heads_last_token(
        backbone=backbone,
        heads=heads,
        tokenizer=tokenizer,
        prompts=["hello"],
        steps=1,
        lr=1e-3,
        device="cpu",
    )

    assert isinstance(loss, float)


def test_distill_heads_all_positions_runs() -> None:
    backbone = DummyBackbone(hidden_size=4, vocab_size=8)
    heads = nn.ModuleList([
        MedusaHead(hidden_size=4, vocab_size=8),
        MedusaHead(hidden_size=4, vocab_size=8),
    ])
    tokenizer = DummyTokenizer()

    loss = distill_heads(
        backbone=backbone,
        heads=heads,
        tokenizer=tokenizer,
        prompts=["hello"],
        steps=2,
        lr=1e-3,
        device="cpu",
    )

    assert isinstance(loss, float)
    assert loss > 0.0
