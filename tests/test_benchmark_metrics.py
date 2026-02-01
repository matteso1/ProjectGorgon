import pytest

from gorgon.benchmarks.metrics import (
    compute_speedup,
    compute_tokens_per_second,
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
