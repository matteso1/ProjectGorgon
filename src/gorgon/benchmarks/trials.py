from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from gorgon.benchmarks.metrics import compute_tokens_per_second, validate_acceptance_rate
from gorgon.benchmarks.report import BenchmarkRun


@dataclass
class TrialResult:
    token_count: int
    elapsed_s: float
    acceptance_rate: Optional[float]


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_trials(name: str, trials: List[TrialResult]) -> BenchmarkRun:
    total_tokens = sum(trial.token_count for trial in trials)
    total_elapsed = sum(trial.elapsed_s for trial in trials)

    acceptance_values = [
        trial.acceptance_rate for trial in trials if trial.acceptance_rate is not None
    ]
    acceptance_rate = None
    if acceptance_values:
        acceptance_rate = validate_acceptance_rate(_mean(acceptance_values))

    return BenchmarkRun(
        name=name,
        token_count=total_tokens,
        elapsed_s=total_elapsed,
        tokens_per_second=compute_tokens_per_second(total_tokens, total_elapsed),
        acceptance_rate=acceptance_rate,
    )
