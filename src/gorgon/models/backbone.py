from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from gorgon.models.medusa_heads import MedusaHead


def load_backbone_4bit(
    model_name: str,
    num_heads: int,
    token: str | None = None,
    device_map: str | None = "auto",
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
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    heads = torch.nn.ModuleList(
        [MedusaHead(hidden_size=hidden_size, vocab_size=vocab_size) for _ in range(num_heads)]
    )
    return model, tokenizer, heads
