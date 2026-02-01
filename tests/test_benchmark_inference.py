from __future__ import annotations

from types import SimpleNamespace
from typing import Iterator, List

import torch

from gorgon.benchmarks.inference import (
    run_baseline,
    run_speculative,
    truncate_prompt_ids,
)
from gorgon.benchmarks.trials import TrialResult


class FakeTokenizer:
    def __call__(self, text: str, return_tensors: str) -> dict:
        tokens = [0, 1]
        return {"input_ids": torch.tensor([tokens])}


class FakeOutputs:
    def __init__(self, logits: torch.Tensor, hidden_states: List[torch.Tensor] | None = None):
        self.logits = logits
        self.hidden_states = hidden_states


class FakeModel:
    def __init__(self, logits_by_len: dict[int, torch.Tensor], hidden_size: int = 4):
        first_logits = next(iter(logits_by_len.values()))
        self.config = SimpleNamespace(vocab_size=first_logits.shape[-1])
        self._logits_by_len = logits_by_len
        self._hidden = torch.zeros(1, 1, hidden_size, dtype=first_logits.dtype)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs) -> torch.Tensor:
        extra = torch.full((input_ids.shape[0], max_new_tokens), 2, dtype=input_ids.dtype)
        return torch.cat([input_ids, extra], dim=1)

    def __call__(self, input_ids: torch.Tensor, output_hidden_states: bool = False, **_) -> FakeOutputs:
        logits = self._logits_by_len[input_ids.shape[1]]
        if output_hidden_states:
            return FakeOutputs(logits=logits, hidden_states=[self._hidden])
        return FakeOutputs(logits=logits)


class FakeHead:
    def __init__(self, token_id: int, vocab_size: int = 3):
        self._logits = torch.zeros(1, vocab_size)
        self._logits[0, token_id] = 1.0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._logits


class FakeHeadWithDevice(FakeHead):
    def __init__(self, token_id: int, vocab_size: int = 3, device: str = "cuda:0"):
        super().__init__(token_id, vocab_size=vocab_size)
        self.device = device
        self.moved_to = None
        self.moved_dtype = None

    def to(self, device: torch.device, dtype: torch.dtype | None = None):
        self.moved_to = str(device)
        self.device = str(device)
        self.moved_dtype = dtype
        return self


def _timer(values: List[float]) -> Iterator[float]:
    for value in values:
        yield value


def test_truncate_prompt_ids() -> None:
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    trimmed = truncate_prompt_ids(ids, prompt_max_length=3)
    assert trimmed.tolist() == [[3, 4, 5]]


def test_run_baseline_counts_tokens() -> None:
    tokenizer = FakeTokenizer()
    logits_by_len = {2: torch.zeros(1, 2, 3)}
    model = FakeModel(logits_by_len)
    timer = _timer([0.0, 2.0])

    result = run_baseline(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=3,
        prompt_max_length=8,
        device="cpu",
        timer=lambda: next(timer),
    )

    assert isinstance(result, TrialResult)
    assert result.token_count == 3
    assert result.elapsed_s == 2.0


def test_run_speculative_accepts_draft() -> None:
    tokenizer = FakeTokenizer()
    vocab_size = 3
    logits_prompt = torch.zeros(1, 2, vocab_size)
    logits_verify = torch.zeros(1, 4, vocab_size)
    logits_verify[0, 1, 1] = 1.0
    logits_verify[0, 2, 2] = 1.0
    model = FakeModel({2: logits_prompt, 4: logits_verify})
    heads = [FakeHead(1, vocab_size), FakeHead(2, vocab_size)]
    timer = _timer([0.0, 4.0])

    result = run_speculative(
        model=model,
        tokenizer=tokenizer,
        heads=heads,
        prompt="hello",
        max_new_tokens=2,
        prompt_max_length=8,
        num_medusa_heads=2,
        device="cpu",
        timer=lambda: next(timer),
    )

    assert result.token_count == 2
    assert result.elapsed_s == 4.0
    assert result.acceptance_rate == 1.0


def test_run_speculative_moves_heads_to_hidden_device() -> None:
    tokenizer = FakeTokenizer()
    vocab_size = 3
    logits_prompt = torch.zeros(1, 2, vocab_size)
    logits_verify = torch.zeros(1, 3, vocab_size)
    logits_verify[0, 1, 1] = 1.0
    model = FakeModel({2: logits_prompt, 3: logits_verify})
    heads = [FakeHeadWithDevice(1, vocab_size, device="cuda:0")]
    timer = _timer([0.0, 1.0])

    result = run_speculative(
        model=model,
        tokenizer=tokenizer,
        heads=heads,
        prompt="hello",
        max_new_tokens=1,
        prompt_max_length=8,
        num_medusa_heads=1,
        device="cpu",
        timer=lambda: next(timer),
    )

    assert heads[0].moved_to == "cuda:0"
    assert result.acceptance_rate == 1.0


def test_run_speculative_casts_head_dtype() -> None:
    tokenizer = FakeTokenizer()
    vocab_size = 3
    logits_prompt = torch.zeros(1, 2, vocab_size, dtype=torch.bfloat16)
    logits_verify = torch.zeros(1, 3, vocab_size, dtype=torch.bfloat16)
    logits_verify[0, 1, 1] = 1.0
    model = FakeModel({2: logits_prompt, 3: logits_verify})
    heads = [FakeHeadWithDevice(1, vocab_size, device="cuda:0")]
    timer = _timer([0.0, 1.0])

    result = run_speculative(
        model=model,
        tokenizer=tokenizer,
        heads=heads,
        prompt="hello",
        max_new_tokens=1,
        prompt_max_length=8,
        num_medusa_heads=1,
        device="cpu",
        timer=lambda: next(timer),
    )

    assert heads[0].moved_dtype == torch.bfloat16
    assert result.acceptance_rate == 1.0
