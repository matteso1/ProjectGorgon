from __future__ import annotations

from gorgon.benchmarks.config import BenchmarkConfig, normalize_prompts
from gorgon.benchmarks.execute import run_prompt_trials, run_warmup
from gorgon.benchmarks.inference import run_baseline, run_speculative
from gorgon.benchmarks.trials import TrialResult, aggregate_trials


def _build_baseline_fn(model, tokenizer, config: BenchmarkConfig, device: str):
    def _run(prompt: str) -> TrialResult:
        return run_baseline(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
            prompt_max_length=config.prompt_max_length,
            device=device,
        )

    return _run


def _build_speculative_fn(model, tokenizer, heads, config: BenchmarkConfig, device: str):
    def _run(prompt: str) -> TrialResult:
        return run_speculative(
            model=model,
            tokenizer=tokenizer,
            heads=heads,
            prompt=prompt,
            max_new_tokens=config.max_new_tokens,
            prompt_max_length=config.prompt_max_length,
            num_medusa_heads=config.num_medusa_heads,
            device=device,
        )

    return _run


def run_benchmark_trials(
    model,
    tokenizer,
    heads,
    config: BenchmarkConfig,
    device: str,
) -> tuple:
    prompts = normalize_prompts(config.prompts)
    baseline_fn = _build_baseline_fn(model, tokenizer, config, device)
    speculative_fn = _build_speculative_fn(model, tokenizer, heads, config, device)

    run_warmup(prompts, config.warmup_steps, baseline_fn)
    run_warmup(prompts, config.warmup_steps, speculative_fn)

    baseline_trials = run_prompt_trials(prompts, config.num_trials, baseline_fn)
    speculative_trials = run_prompt_trials(prompts, config.num_trials, speculative_fn)

    baseline_run = aggregate_trials("baseline", baseline_trials)
    speculative_run = aggregate_trials("speculative", speculative_trials)

    return baseline_run, speculative_run
