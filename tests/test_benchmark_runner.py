import pytest

from gorgon.benchmarks.runner import (
    ensure_hf_token,
    format_run_name,
    infer_device,
    select_head_train_device,
)


def test_infer_device_prefers_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gorgon.benchmarks.runner.torch.cuda.is_available", lambda: True)
    assert infer_device("auto") == "cuda"


def test_infer_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gorgon.benchmarks.runner.torch.cuda.is_available", lambda: False)
    assert infer_device("auto") == "cpu"


def test_infer_device_respects_explicit() -> None:
    assert infer_device("cpu") == "cpu"
    assert infer_device("cuda") == "cuda"


def test_ensure_hf_token_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "abc123")
    assert ensure_hf_token() == "abc123"


def test_ensure_hf_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        ensure_hf_token()


def test_format_run_name() -> None:
    assert format_run_name("baseline", 3) == "baseline/trial-3"


def test_select_head_train_device_defaults_to_cpu_on_cuda() -> None:
    assert select_head_train_device("cuda") == "cpu"
    assert select_head_train_device("cpu") == "cpu"
