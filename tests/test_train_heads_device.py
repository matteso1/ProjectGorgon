import torch

from gorgon.training.train_heads import move_targets_to_device


def test_move_targets_to_device_moves_tensors() -> None:
    targets = [
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[5, 6], [7, 8]]),
    ]

    moved = move_targets_to_device(targets, device=torch.device("cpu"))

    assert all(t.device.type == "cpu" for t in moved)
    assert moved[0].shape == targets[0].shape
