from gorgon.benchmarks.config import BenchmarkConfig, default_prompts, normalize_prompts


def test_default_prompts_returns_copy() -> None:
    prompts = default_prompts()
    assert prompts == [
        "The quick brown fox jumps over the lazy dog.",
        "Explain the difference between TCP and UDP in one paragraph.",
        "Write a short story about a robot learning to paint.",
        "Summarize the plot of Hamlet in two sentences.",
    ]

    prompts.append("mutate")
    assert default_prompts() == [
        "The quick brown fox jumps over the lazy dog.",
        "Explain the difference between TCP and UDP in one paragraph.",
        "Write a short story about a robot learning to paint.",
        "Summarize the plot of Hamlet in two sentences.",
    ]


def test_normalize_prompts_strips_and_drops_empty() -> None:
    assert normalize_prompts(["  hi  ", "", "  ", "\tthere"]) == ["hi", "there"]


def test_benchmark_config_defaults() -> None:
    cfg = BenchmarkConfig()

    assert cfg.model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert cfg.max_new_tokens == 128
    assert cfg.prompt_max_length == 256
    assert cfg.warmup_steps == 2
    assert cfg.num_trials == 5
    assert cfg.seed == 0
    assert cfg.num_medusa_heads == 4
    assert cfg.top_k == 4
