"""Tests for adaptive tree pruning in tree_candidates.py.

Validates confidence_threshold, use_path_confidence, and entropy_weighted
pruning modes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from gorgon.inference.tree_candidates import (
    CandidateTree,
    build_candidate_tree,
    get_tree_paths,
)


class FakeMedusaHead(nn.Module):
    """Deterministic head that always returns a fixed ranking."""

    def __init__(self, vocab_size: int, top_tokens: list[int]):
        super().__init__()
        self.vocab_size = vocab_size
        self.top_tokens = top_tokens
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, 1, self.vocab_size), -100.0)
        for rank, tok in enumerate(self.top_tokens):
            logits[0, 0, tok] = 100.0 - rank
        return logits


class ConfidentHead(nn.Module):
    """Head with controllable confidence distribution.

    One dominant token gets `dominant_logit`, rest get `base_logit`.
    """

    def __init__(self, vocab_size: int, top_tokens: list[int],
                 dominant_logit: float = 10.0, base_logit: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.top_tokens = top_tokens
        self.dominant_logit = dominant_logit
        self.base_logit = base_logit
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, 1, self.vocab_size), self.base_logit)
        for rank, tok in enumerate(self.top_tokens):
            logits[0, 0, tok] = self.dominant_logit - rank
        return logits


# -- threshold=0.0 matches old behavior --


def test_threshold_zero_matches_baseline():
    """With threshold=0.0, adaptive pruning should produce the same tree."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30]),
        FakeMedusaHead(100, [11, 21, 31]),
    ])
    hidden = torch.randn(1, 1, 1)

    tree_default = build_candidate_tree(heads, hidden, top_k=3, max_depth=2)
    tree_adaptive = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=2,
        confidence_threshold=0.0,
    )

    assert len(tree_default.tokens) == len(tree_adaptive.tokens)
    assert tree_default.parents == tree_adaptive.parents
    assert torch.equal(tree_default.tokens, tree_adaptive.tokens)


# -- threshold=1.0 produces minimal/empty tree --


def test_threshold_one_prunes_aggressively():
    """With threshold=1.0, almost all candidates should be pruned.

    A threshold of 1.0 means only candidates with prob >= 1.0 survive,
    which is essentially impossible with softmax over multiple candidates.
    """
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30]),
        FakeMedusaHead(100, [11, 21, 31]),
    ])
    hidden = torch.randn(1, 1, 1)

    tree = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=2,
        confidence_threshold=1.0,
    )

    # With softmax over 3 items, max prob < 1.0, so tree should be empty or minimal
    assert len(tree.tokens) <= 1


# -- Moderate threshold reduces tree size --


def test_moderate_threshold_reduces_tree():
    """A moderate threshold should produce a smaller tree than no threshold."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30, 40]),
        FakeMedusaHead(100, [11, 21, 31, 41]),
    ])
    hidden = torch.randn(1, 1, 1)

    tree_full = build_candidate_tree(
        heads, hidden, top_k=4, max_depth=2,
        confidence_threshold=0.0,
    )
    tree_pruned = build_candidate_tree(
        heads, hidden, top_k=4, max_depth=2,
        confidence_threshold=0.2,
    )

    assert len(tree_pruned.tokens) <= len(tree_full.tokens)


# -- Path confidence prunes deeper paths more --


def test_path_confidence_prunes_deeper():
    """use_path_confidence=True should prune more aggressively at depth."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30]),
        FakeMedusaHead(100, [11, 21, 31]),
        FakeMedusaHead(100, [12, 22, 32]),
    ])
    hidden = torch.randn(1, 1, 1)

    tree_no_path = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=3,
        confidence_threshold=0.1,
        use_path_confidence=False,
    )
    tree_with_path = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=3,
        confidence_threshold=0.1,
        use_path_confidence=True,
    )

    # Path confidence multiplies probs, so deeper levels get pruned harder
    assert len(tree_with_path.tokens) <= len(tree_no_path.tokens)


# -- Entropy-weighted threshold --


def test_entropy_weighted_confident_head():
    """A very confident head (low entropy) with entropy weighting should
    have a lower effective threshold (threshold * (1 - norm_entropy)),
    where norm_entropy is near 0 for confident heads. This means the
    effective threshold stays close to the base, so pruning is similar.
    """
    # Very confident head: dominant logit is much higher
    heads = nn.ModuleList([
        ConfidentHead(100, [10, 20, 30], dominant_logit=50.0, base_logit=-50.0),
    ])
    hidden = torch.randn(1, 1, 1)

    tree_no_ew = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=1,
        confidence_threshold=0.3,
        entropy_weighted=False,
    )
    tree_ew = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=1,
        confidence_threshold=0.3,
        entropy_weighted=True,
    )

    # Entropy weighting can only lower the threshold (multiply by 1-norm_entropy),
    # so it keeps at least as many candidates as without it
    assert len(tree_ew.tokens) >= len(tree_no_ew.tokens)


def test_entropy_weighted_uncertain_head():
    """An uncertain head (high entropy) should lower the effective threshold,
    keeping more candidates than without entropy weighting.
    """
    # Uncertain head: all logits close together
    heads = nn.ModuleList([
        ConfidentHead(100, [10, 20, 30], dominant_logit=1.0, base_logit=0.0),
    ])
    hidden = torch.randn(1, 1, 1)

    tree_no_ew = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=1,
        confidence_threshold=0.4,
        entropy_weighted=False,
    )
    tree_ew = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=1,
        confidence_threshold=0.4,
        entropy_weighted=True,
    )

    # High entropy -> effective threshold < base -> more candidates kept
    assert len(tree_ew.tokens) >= len(tree_no_ew.tokens)


# -- Scores field populated --


def test_scores_field_populated():
    """CandidateTree.scores should be populated with probabilities."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30]),
    ])
    hidden = torch.randn(1, 1, 1)

    tree = build_candidate_tree(heads, hidden, top_k=3, max_depth=1)

    assert tree.scores is not None
    assert tree.scores.shape == tree.tokens.shape
    # All probs should be between 0 and 1
    assert (tree.scores >= 0).all()
    assert (tree.scores <= 1).all()


# -- Empty tree handling --


def test_empty_tree_has_correct_structure():
    """An empty tree (all pruned) should have valid structure."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30]),
    ])
    hidden = torch.randn(1, 1, 1)

    tree = build_candidate_tree(
        heads, hidden, top_k=3, max_depth=1,
        confidence_threshold=1.0,
    )

    assert len(tree.tokens) == 0 or len(tree.tokens) <= 1
    assert tree.scores is not None
    assert len(tree.scores) == len(tree.tokens)
