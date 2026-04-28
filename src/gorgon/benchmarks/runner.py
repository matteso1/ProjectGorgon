from __future__ import annotations

import os

import torch


def ensure_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to download the model.")
    return token


def infer_device(preference: str) -> str:
    if preference in {"cpu", "cuda"}:
        return preference
    return "cuda" if torch.cuda.is_available() else "cpu"


def format_run_name(kind: str, trial: int) -> str:
    return f"{kind}/trial-{trial}"


def select_head_train_device(device: str) -> str:
    if device == "cuda":
        return "cpu"
    return device
