import torch


def test_cuda_visible():
    assert torch.cuda.is_available(), "CUDA must be available in WSL2"
