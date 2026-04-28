from __future__ import annotations

import platform

import torch

from gorgon.benchmarks.report import SystemInfo


def gather_system_info() -> SystemInfo:
    cuda_version = torch.version.cuda or "unknown"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    else:
        gpu = "cpu"
        cuda_version = "unknown"

    return SystemInfo(
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        gpu=gpu,
    )
