from gorgon.benchmarks.entrypoint import build_config_summary
from gorgon.benchmarks.config import BenchmarkConfig


def test_build_config_summary() -> None:
    cfg = BenchmarkConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=64,
        prompt_max_length=128,
        warmup_steps=1,
        num_trials=2,
        seed=7,
        num_medusa_heads=6,
        top_k=8,
    )

    summary = build_config_summary(cfg)

    assert summary.model_name == cfg.model_name
    assert summary.max_new_tokens == 64
    assert summary.prompt_max_length == 128
    assert summary.warmup_steps == 1
    assert summary.num_trials == 2
    assert summary.seed == 7
    assert summary.num_medusa_heads == 6
    assert summary.top_k == 8
