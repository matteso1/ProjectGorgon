import torch

from gorgon.models.medusa_heads import MedusaHead


def test_medusa_head_output_shape():
    head = MedusaHead(hidden_size=8, vocab_size=16)
    x = torch.randn(2, 4, 8)
    y = head(x)
    assert y.shape == (2, 4, 16)
