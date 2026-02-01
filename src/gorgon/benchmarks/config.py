from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


_DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Explain the difference between TCP and UDP in one paragraph.",
    "Write a short story about a robot learning to paint.",
    "Summarize the plot of Hamlet in two sentences.",
]


def default_prompts() -> List[str]:
    return list(_DEFAULT_PROMPTS)


def normalize_prompts(prompts: List[str]) -> List[str]:
    return [p.strip() for p in prompts if p.strip()]


@dataclass
class BenchmarkConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_new_tokens: int = 128
    prompt_max_length: int = 256
    warmup_steps: int = 2
    num_trials: int = 5
    seed: int = 0
    num_medusa_heads: int = 4
    top_k: int = 4
    prompts: List[str] = field(default_factory=default_prompts)
