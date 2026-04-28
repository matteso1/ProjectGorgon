from gorgon.benchmarks.trials import TrialResult, aggregate_trials


def test_aggregate_trials_sums_and_averages() -> None:
    trials = [
        TrialResult(token_count=10, elapsed_s=2.0, acceptance_rate=0.4),
        TrialResult(token_count=20, elapsed_s=3.0, acceptance_rate=0.6),
    ]

    run = aggregate_trials("speculative", trials)

    assert run.name == "speculative"
    assert run.token_count == 30
    assert run.elapsed_s == 5.0
    assert run.tokens_per_second == 6.0
    assert run.acceptance_rate == 0.5


def test_aggregate_trials_handles_missing_acceptance() -> None:
    trials = [
        TrialResult(token_count=10, elapsed_s=2.0, acceptance_rate=None),
        TrialResult(token_count=20, elapsed_s=3.0, acceptance_rate=None),
    ]

    run = aggregate_trials("baseline", trials)

    assert run.name == "baseline"
    assert run.token_count == 30
    assert run.elapsed_s == 5.0
    assert run.tokens_per_second == 6.0
    assert run.acceptance_rate is None
