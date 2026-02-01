from __future__ import annotations

from gorgon.benchmarks.config import BenchmarkConfig
from gorgon.benchmarks.report import BenchmarkConfigSummary


def build_config_summary(config: BenchmarkConfig) -> BenchmarkConfigSummary:
    return BenchmarkConfigSummary(
        model_name=config.model_name,
        max_new_tokens=config.max_new_tokens,
        prompt_max_length=config.prompt_max_length,
        warmup_steps=config.warmup_steps,
        num_trials=config.num_trials,
        seed=config.seed,
        num_medusa_heads=config.num_medusa_heads,
        top_k=config.top_k,
    )
