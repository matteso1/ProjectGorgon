from __future__ import annotations

from typing import Dict, List, Optional


def compute_tokens_per_second(token_count: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return token_count / elapsed_s


def compute_speedup(baseline_tps: float, candidate_tps: float) -> float:
    if baseline_tps <= 0:
        return 0.0
    return candidate_tps / baseline_tps


def validate_acceptance_rate(rate: Optional[float]) -> Optional[float]:
    if rate is None:
        return None
    if not 0.0 <= rate <= 1.0:
        raise ValueError("acceptance_rate must be between 0 and 1")
    return rate


def compute_mean_accepted_length(accepted_lengths: List[int]) -> float:
    """Compute mean accepted tokens per iteration (tau)."""
    if not accepted_lengths:
        return 0.0
    return sum(accepted_lengths) / len(accepted_lengths)


def compute_per_head_acceptance(
    head_acceptances: List[List[bool]],
) -> List[float]:
    """Compute per-head acceptance rate across iterations."""
    if not head_acceptances:
        return []
    max_depth = max(len(ha) for ha in head_acceptances)
    rates: List[float] = []
    for h in range(max_depth):
        total = 0
        accepted = 0
        for ha in head_acceptances:
            if h < len(ha):
                total += 1
                if ha[h]:
                    accepted += 1
        rates.append(accepted / total if total > 0 else 0.0)
    return rates


def compute_tree_utilization(
    accepted_lengths: List[int], tree_sizes: List[int]
) -> float:
    """Fraction of tree nodes accepted on average."""
    if not accepted_lengths or not tree_sizes:
        return 0.0
    utils = []
    for acc, size in zip(accepted_lengths, tree_sizes):
        if size > 0:
            utils.append(acc / size)
    return sum(utils) / len(utils) if utils else 0.0


def summarize_run(
    name: str,
    token_count: int,
    elapsed_s: float,
    acceptance_rate: Optional[float],
) -> Dict[str, float | int | str | None]:
    return {
        "name": name,
        "token_count": token_count,
        "elapsed_s": elapsed_s,
        "tokens_per_second": compute_tokens_per_second(token_count, elapsed_s),
        "acceptance_rate": validate_acceptance_rate(acceptance_rate),
    }
