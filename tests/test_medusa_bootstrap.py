from types import SimpleNamespace

import torch
import torch.nn as nn

from gorgon.models.medusa_heads import MedusaHead
from gorgon.training.medusa_bootstrap import build_head_targets, train_heads_on_prompts


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, output_hidden_states, return_dict):
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.hidden_size)
        return SimpleNamespace(hidden_states=[hidden])


class DummyTokenizer:
    def __call__(self, text: str, return_tensors: str):
        return {"input_ids": torch.tensor([[1, 2, 3, 4]])}


def test_build_head_targets_shapes() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4]])
    targets = build_head_targets(input_ids, max_heads=2, ignore_index=-100)

    assert len(targets) == 2
    assert targets[0].shape == (1, 4)
    assert targets[1].shape == (1, 4)


def test_train_heads_on_prompts_runs_step() -> None:
    backbone = DummyBackbone(hidden_size=4)
    heads = nn.ModuleList([MedusaHead(hidden_size=4, vocab_size=8)])
    tokenizer = DummyTokenizer()

    loss = train_heads_on_prompts(
        backbone=backbone,
        heads=heads,
        tokenizer=tokenizer,
        prompts=["hello"],
        steps=1,
        lr=1e-3,
        device="cpu",
    )

    assert isinstance(loss, float)
