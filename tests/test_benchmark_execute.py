from gorgon.benchmarks.execute import run_prompt_trials, run_warmup
from gorgon.benchmarks.trials import TrialResult


def test_run_prompt_trials_calls_all_prompts() -> None:
    calls = []

    def run_fn(prompt: str) -> TrialResult:
        calls.append(prompt)
        return TrialResult(token_count=1, elapsed_s=1.0, acceptance_rate=None)

    prompts = ["a", "b"]
    results = run_prompt_trials(prompts, num_trials=2, run_fn=run_fn)

    assert calls == ["a", "b", "a", "b"]
    assert len(results) == 4


def test_run_warmup_calls_all_prompts() -> None:
    calls = []

    def run_fn(prompt: str) -> TrialResult:
        calls.append(prompt)
        return TrialResult(token_count=1, elapsed_s=1.0, acceptance_rate=None)

    prompts = ["x", "y", "z"]
    run_warmup(prompts, warmup_steps=2, run_fn=run_fn)

    assert calls == ["x", "y", "z", "x", "y", "z"]
