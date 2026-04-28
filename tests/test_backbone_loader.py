from types import SimpleNamespace

import torch
import torch.nn as nn

from gorgon.models import backbone as backbone_mod


def test_load_backbone_passes_device_map_and_low_cpu(monkeypatch) -> None:
    captured = {}

    lm_head = nn.Linear(4, 8, bias=False)

    def fake_from_pretrained(model_name, **kwargs):
        captured.update(kwargs)
        fake_model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=4, vocab_size=8),
            parameters=lambda: [],
            lm_head=lm_head,
        )
        return fake_model

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        return SimpleNamespace(vocab_size=8)

    monkeypatch.setattr(backbone_mod.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(backbone_mod.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)

    _, _, heads = backbone_mod.load_backbone_4bit(
        model_name="test-model",
        num_heads=1,
        token="token",
        device_map="cuda",
    )

    assert captured["device_map"] == "cuda"
    assert captured["low_cpu_mem_usage"] is True

    # Verify lm_head weight was copied
    assert torch.allclose(heads[0].lm_head.weight.data, lm_head.weight.data)
