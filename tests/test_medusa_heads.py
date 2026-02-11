import torch

from gorgon.models.medusa_heads import MedusaHead


def test_medusa_head_output_shape():
    head = MedusaHead(hidden_size=8, vocab_size=16)
    x = torch.randn(2, 4, 8)
    y = head(x)
    assert y.shape == (2, 4, 16)


def test_medusa_head_weights_update_after_step():
    torch.manual_seed(0)
    head = MedusaHead(hidden_size=4, vocab_size=6)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)

    inputs = torch.tensor(
        [
            [1.0, -1.0, 0.5, 2.0],
            [-0.5, 1.5, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 2], dtype=torch.long)

    first = head.blocks[0].linear
    second = head.lm_head
    first_before = first.weight.detach().clone()
    second_before = second.weight.detach().clone()

    logits = head(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()

    assert not torch.allclose(first.weight, first_before)
    assert not torch.allclose(second.weight, second_before)


def test_medusa_head_residual_shape_preservation():
    """With multiple residual blocks, shape is preserved throughout."""
    head = MedusaHead(hidden_size=8, vocab_size=16, num_residual_blocks=2)
    x = torch.randn(2, 4, 8)
    y = head(x)
    assert y.shape == (2, 4, 16)
    assert len(head.blocks) == 2
