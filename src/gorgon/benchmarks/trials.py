from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gorgon.benchmarks.metrics import (
    compute_mean_accepted_length,
    compute_per_head_acceptance,
    compute_tokens_per_second,
    compute_tree_utilization,
    validate_acceptance_rate,
)
from gorgon.benchmarks.report import BenchmarkRun


@dataclass
class TrialResult:
    token_count: int
    elapsed_s: float
    acceptance_rate: Optional[float]
    mean_accepted_length: Optional[float] = None
    per_head_acceptance: Optional[List[float]] = None
    tree_utilization: Optional[float] = None
    time_breakdown: Optional[Dict[str, float]] = None


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

    # Aggregate new metrics
    tau_values = [
        trial.mean_accepted_length
        for trial in trials
        if trial.mean_accepted_length is not None
    ]
    mean_accepted_length = _mean(tau_values) if tau_values else None

    util_values = [
        trial.tree_utilization
        for trial in trials
        if trial.tree_utilization is not None
    ]
    tree_utilization = _mean(util_values) if util_values else None

    # Average per-head acceptance across trials
    head_acc_lists = [
        trial.per_head_acceptance
        for trial in trials
        if trial.per_head_acceptance is not None
    ]
    per_head_acceptance = None
    if head_acc_lists:
        max_heads = max(len(ha) for ha in head_acc_lists)
        per_head_acceptance = []
        for h in range(max_heads):
            vals = [ha[h] for ha in head_acc_lists if h < len(ha)]
            per_head_acceptance.append(_mean(vals))

    # Average time breakdown
    time_breakdowns = [
        trial.time_breakdown
        for trial in trials
        if trial.time_breakdown is not None
    ]
    time_breakdown = None
    if time_breakdowns:
        all_keys = set()
        for tb in time_breakdowns:
            all_keys.update(tb.keys())
        time_breakdown = {}
        for key in sorted(all_keys):
            vals = [tb[key] for tb in time_breakdowns if key in tb]
            time_breakdown[key] = _mean(vals)

    return BenchmarkRun(
        name=name,
        token_count=total_tokens,
        elapsed_s=total_elapsed,
        tokens_per_second=compute_tokens_per_second(total_tokens, total_elapsed),
        acceptance_rate=acceptance_rate,
        mean_accepted_length=mean_accepted_length,
        per_head_acceptance=per_head_acceptance,
        tree_utilization=tree_utilization,
        time_breakdown=time_breakdown,
    )
