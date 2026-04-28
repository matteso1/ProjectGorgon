import pytest
import torch

from gorgon.inference.gorgon_loop import (
    accept_draft_tokens,
    IterationStats,
    SpeculativeResult,
)


def test_accept_draft_tokens_greedy_rejects_first_mismatch():
    draft = [1, 2, 3]
    logits = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0],  # argmax -> 1
            [0.0, 0.0, 3.0, 0.0],  # argmax -> 2
            [4.0, 0.0, 0.0, 0.0],  # argmax -> 0 (mismatch)
        ],
        dtype=torch.float32,
    )

    accepted, rejected_at = accept_draft_tokens(draft, logits)

    assert accepted == [1, 2]
    assert rejected_at == 2


# --- IterationStats and SpeculativeResult computed properties ---


def test_iteration_stats_dataclass():
    stats = IterationStats(
        tree_size=12,
        accepted_length=3,
        head_acceptance=[True, True, True],
        time_draft_ms=1.5,
        time_verify_ms=10.0,
        time_kv_trim_ms=0.2,
    )
    assert stats.tree_size == 12
    assert stats.accepted_length == 3
    assert len(stats.head_acceptance) == 3


def test_speculative_result_mean_accepted_length():
    stats = [
        IterationStats(12, 2, [True, True, False], 1.0, 5.0, 0.1),
        IterationStats(12, 4, [True, True, True, True], 1.0, 5.0, 0.1),
    ]
    result = SpeculativeResult(
        generated_ids=[1, 2, 3],
        total_draft_tokens=24,
        total_accepted_tokens=6,
        num_iterations=2,
        iteration_stats=stats,
    )
    assert result.mean_accepted_length == pytest.approx(3.0)


def test_speculative_result_per_head_acceptance_rates():
    stats = [
        IterationStats(12, 2, [True, True, False], 1.0, 5.0, 0.1),
        IterationStats(12, 1, [True, False, False], 1.0, 5.0, 0.1),
    ]
    result = SpeculativeResult(
        generated_ids=[1, 2, 3],
        total_draft_tokens=24,
        total_accepted_tokens=3,
        num_iterations=2,
        iteration_stats=stats,
    )
    rates = result.per_head_acceptance_rates
    assert len(rates) == 3
    assert rates[0] == pytest.approx(1.0)
    assert rates[1] == pytest.approx(0.5)
    assert rates[2] == pytest.approx(0.0)


def test_speculative_result_tree_utilization():
    stats = [
        IterationStats(10, 2, [True, True], 1.0, 5.0, 0.1),
        IterationStats(10, 4, [True, True, True, True], 1.0, 5.0, 0.1),
    ]
    result = SpeculativeResult(
        generated_ids=[1, 2, 3],
        total_draft_tokens=20,
        total_accepted_tokens=6,
        num_iterations=2,
        iteration_stats=stats,
    )
    # (2/10 + 4/10) / 2 = 0.3
    assert result.tree_utilization == pytest.approx(0.3)


def test_speculative_result_empty_stats():
    result = SpeculativeResult(
        generated_ids=[],
        total_draft_tokens=0,
        total_accepted_tokens=0,
        num_iterations=0,
    )
    assert result.mean_accepted_length == 0.0
    assert result.per_head_acceptance_rates == []
    assert result.tree_utilization == 0.0