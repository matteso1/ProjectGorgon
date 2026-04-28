from types import SimpleNamespace

import torch
import torch.nn as nn

from gorgon.data.dataset import make_shifted_targets
from gorgon.models.medusa_heads import MedusaHead
from gorgon.training.train_heads import replace_none_with_ignore, train_step


class DummyBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, output_hidden_states=True, return_dict=True):
        hidden = self.embed(input_ids)
        return SimpleNamespace(hidden_states=(hidden,))


def test_train_heads_overfit():
    torch.manual_seed(0)
    vocab_size = 32
    hidden_size = 16
    backbone = DummyBackbone(vocab_size=vocab_size, hidden_size=hidden_size)
    for param in backbone.parameters():
        param.requires_grad = False

    heads = nn.ModuleList(
        [
            MedusaHead(hidden_size=hidden_size, vocab_size=vocab_size),
            MedusaHead(hidden_size=hidden_size, vocab_size=vocab_size),
        ]
    )
    optimizer = torch.optim.AdamW(heads.parameters(), lr=0.2)

    tokens = [1, 2, 3, 4, 5]
    input_ids = torch.tensor([tokens[:-1], tokens[:-1]])
    t1, t2 = make_shifted_targets(tokens, max_heads=2)
    target_ids = [
        replace_none_with_ignore([t1, t1]),
        replace_none_with_ignore([t2, t2]),
    ]

    losses = []
    for _ in range(25):
        loss = train_step(
            backbone=backbone,
            heads=heads,
            input_ids=input_ids,
            target_ids=target_ids,
            optimizer=optimizer,
        )
        losses.append(loss)

    assert losses[-1] < losses[0]
