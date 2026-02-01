from gorgon.benchmarks.config import BenchmarkConfig
from gorgon.benchmarks.runner_core import run_benchmark_trials


class Dummy:
    pass


def test_run_benchmark_trials_returns_runs(monkeypatch) -> None:
    config = BenchmarkConfig(
        max_new_tokens=2,
        prompt_max_length=8,
        warmup_steps=1,
        num_trials=1,
    )
    config.prompts = ["a"]

    monkeypatch.setattr(
        "gorgon.benchmarks.runner_core.run_baseline",
        lambda *args, **kwargs: type("T", (), {"token_count": 2, "elapsed_s": 2.0, "acceptance_rate": None})(),
    )
    monkeypatch.setattr(
        "gorgon.benchmarks.runner_core.run_speculative",
        lambda *args, **kwargs: type("T", (), {"token_count": 2, "elapsed_s": 1.0, "acceptance_rate": 1.0})(),
    )

    baseline, speculative = run_benchmark_trials(Dummy(), Dummy(), Dummy(), config, device="cpu")

    assert baseline.name == "baseline"
    assert speculative.name == "speculative"
