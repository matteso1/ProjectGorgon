import pytest

from gorgon.benchmarks.metrics import (
    compute_mean_accepted_length,
    compute_per_head_acceptance,
    compute_speedup,
    compute_tokens_per_second,
    compute_tree_utilization,
    summarize_run,
    validate_acceptance_rate,
)


def test_compute_tokens_per_second_handles_zero_elapsed() -> None:
    assert compute_tokens_per_second(10, 0.0) == 0.0


def test_compute_tokens_per_second_basic() -> None:
    assert compute_tokens_per_second(10, 2.0) == pytest.approx(5.0)


def test_compute_speedup_handles_zero_baseline() -> None:
    assert compute_speedup(0.0, 10.0) == 0.0


def test_compute_speedup_basic() -> None:
    assert compute_speedup(2.0, 10.0) == pytest.approx(5.0)


def test_validate_acceptance_rate() -> None:
    assert validate_acceptance_rate(None) is None
    assert validate_acceptance_rate(0.0) == 0.0
    assert validate_acceptance_rate(1.0) == 1.0

    with pytest.raises(ValueError):
        validate_acceptance_rate(-0.1)

    with pytest.raises(ValueError):
        validate_acceptance_rate(1.01)


def test_summarize_run() -> None:
    summary = summarize_run("baseline", 20, 2.0, None)

    assert summary["name"] == "baseline"
    assert summary["token_count"] == 20
    assert summary["elapsed_s"] == pytest.approx(2.0)
    assert summary["tokens_per_second"] == pytest.approx(10.0)
    assert summary["acceptance_rate"] is None


# --- New metric functions ---


def test_compute_mean_accepted_length_empty() -> None:
    assert compute_mean_accepted_length([]) == 0.0


def test_compute_mean_accepted_length_basic() -> None:
    assert compute_mean_accepted_length([2, 4, 3]) == pytest.approx(3.0)


def test_compute_per_head_acceptance_empty() -> None:
    assert compute_per_head_acceptance([]) == []


def test_compute_per_head_acceptance_basic() -> None:
    acceptances = [
        [True, True, False],
        [True, False, False],
        [True, True, True],
    ]
    rates = compute_per_head_acceptance(acceptances)
    assert len(rates) == 3
    assert rates[0] == pytest.approx(1.0)
    assert rates[1] == pytest.approx(2.0 / 3.0)
    assert rates[2] == pytest.approx(1.0 / 3.0)


def test_compute_tree_utilization_empty() -> None:
    assert compute_tree_utilization([], []) == 0.0


def test_compute_tree_utilization_basic() -> None:
    # 2 accepted out of 10, 4 accepted out of 8 -> avg(0.2, 0.5) = 0.35
    util = compute_tree_utilization([2, 4], [10, 8])
    assert util == pytest.approx(0.35)
