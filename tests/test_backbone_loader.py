from types import SimpleNamespace

from gorgon.models import backbone as backbone_mod


def test_load_backbone_passes_device_map_and_low_cpu(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(config=SimpleNamespace(hidden_size=4, vocab_size=8), parameters=lambda: [])

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        return SimpleNamespace(vocab_size=8)

    monkeypatch.setattr(backbone_mod.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(backbone_mod.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)

    backbone_mod.load_backbone_4bit(
        model_name="test-model",
        num_heads=1,
        token="token",
        device_map="cuda",
    )

    assert captured["device_map"] == "cuda"
    assert captured["low_cpu_mem_usage"] is True
