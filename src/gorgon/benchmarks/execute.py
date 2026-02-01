from __future__ import annotations

from typing import Callable, Iterable, List

from gorgon.benchmarks.trials import TrialResult


def run_warmup(
    prompts: Iterable[str],
    warmup_steps: int,
    run_fn: Callable[[str], TrialResult],
) -> None:
    for _ in range(warmup_steps):
        for prompt in prompts:
            run_fn(prompt)


def run_prompt_trials(
    prompts: Iterable[str],
    num_trials: int,
    run_fn: Callable[[str], TrialResult],
) -> List[TrialResult]:
    results: List[TrialResult] = []
    for _ in range(num_trials):
        for prompt in prompts:
            results.append(run_fn(prompt))
    return results
