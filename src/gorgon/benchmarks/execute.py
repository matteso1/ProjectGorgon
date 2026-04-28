from __future__ import annotations

from typing import Callable, Iterable, List

from gorgon.benchmarks.trials import TrialResult
from tqdm import tqdm


def run_warmup(
    prompts: Iterable[str],
    warmup_steps: int,
    run_fn: Callable[[str], TrialResult],
) -> None:
    if warmup_steps <= 0:
        return
    prompt_list = list(prompts)
    for _ in tqdm(range(warmup_steps), desc="warmup", unit="pass"):
        for prompt in prompt_list:
            run_fn(prompt)


def run_prompt_trials(
    prompts: Iterable[str],
    num_trials: int,
    run_fn: Callable[[str], TrialResult],
) -> List[TrialResult]:
    results: List[TrialResult] = []
    if num_trials <= 0:
        return results
    prompt_list = list(prompts)
    for _ in tqdm(range(num_trials), desc="trials", unit="trial"):
        for prompt in prompt_list:
            results.append(run_fn(prompt))
    return results
