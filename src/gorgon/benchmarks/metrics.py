from __future__ import annotations

from typing import Dict, Optional


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
