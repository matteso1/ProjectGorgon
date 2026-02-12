from __future__ import annotations

import copy
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gorgon.models.medusa_heads import MedusaHead


def _get_backbone_norm(model) -> torch.nn.Module | None:
    """Extract the final RMSNorm/LayerNorm from the backbone.

    Llama-family models use model.model.norm (RMSNorm).
    Returns a frozen deep copy, or None if not found.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return None
    norm = getattr(inner, "norm", None)
    if norm is None:
        return None
    norm_copy = copy.deepcopy(norm)
    for param in norm_copy.parameters():
        param.requires_grad = False
    return norm_copy


def load_backbone_4bit(
    model_name: str,
    num_heads: int,
    token: str | None = None,
    device_map: str | None = "auto",
    low_cpu_mem_usage: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.nn.ModuleList]:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size

    # Extract frozen norm layer from backbone for Medusa heads
    backbone_norm = _get_backbone_norm(model)

    heads = torch.nn.ModuleList([
        MedusaHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            norm=copy.deepcopy(backbone_norm) if backbone_norm is not None else None,
        )
        for _ in range(num_heads)
    ])

    # Copy backbone lm_head weights into each Medusa head's lm_head
    if hasattr(model, 'lm_head'):
        with torch.no_grad():
            for head in heads:
                head.lm_head.weight.data.copy_(model.lm_head.weight.data)

    return model, tokenizer, heads


def load_trained_heads(
    checkpoint_path: str | Path,
    heads: torch.nn.ModuleList,
    device: str | torch.device = "cpu",
) -> int:
    """Load trained Medusa head weights from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        heads: ModuleList of MedusaHead instances to load weights into.
        device: Device to map weights to.

    Returns:
        The training step the checkpoint was saved at.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["heads_state_dict"]

    # Handle torch.compile() checkpoints: strip _orig_mod. prefix
    compiled_prefix = "_orig_mod."
    if any(k.split(".", 1)[-1].startswith(compiled_prefix) for k in state_dict):
        state_dict = {
            k.replace(compiled_prefix, ""): v for k, v in state_dict.items()
        }

    # Handle architecture mismatch: old checkpoints without norm layer
    head_keys = set(heads.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = head_keys - ckpt_keys
    norm_missing = [k for k in missing if ".norm." in k]

    if norm_missing and not (missing - set(norm_missing)):
        # Only norm keys are missing -- old checkpoint without norm.
        # Load what we can (strict=False), norm stays at backbone-initialized values.
        heads.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint (step {ckpt['step']}), norm weights kept from backbone init")
    else:
        heads.load_state_dict(state_dict)
        print(f"  Loaded checkpoint (step {ckpt['step']}), loss={ckpt.get('loss', '?')}")

    # Ensure heads are on the correct device and dtype
    if device is not None:
        heads.to(device)
    # Match dtype of loaded weights (checkpoint may be bf16 or fp16)
    first_param = next(iter(state_dict.values()))
    heads.to(first_param.dtype)

    return ckpt["step"]
