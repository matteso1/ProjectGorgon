"""Unit tests for tree candidate generation and verification logic.

These tests validate the correctness of the core speculative decoding
engine WITHOUT requiring a GPU or the full Llama backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gorgon.inference.tree_candidates import (
    CandidateTree,
    build_candidate_tree,
    candidate_tree_to_mask,
    get_tree_paths,
)
from gorgon.inference.gorgon_loop import accept_draft_tokens


# ── Helpers ──────────────────────────────────────────────────────────


class FakeMedusaHead(nn.Module):
    """Deterministic head that always returns a fixed ranking."""

    def __init__(self, vocab_size: int, top_tokens: list[int]):
        super().__init__()
        self.vocab_size = vocab_size
        self.top_tokens = top_tokens
        # Need at least one parameter so `next(heads.parameters())` works
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, 1, self.vocab_size), -100.0)
        for rank, tok in enumerate(self.top_tokens):
            logits[0, 0, tok] = 100.0 - rank
        return logits


# ── Tree structure tests ─────────────────────────────────────────────


def test_tree_size_single_head():
    """1 head with top-k=4 → 4 candidates."""
    heads = nn.ModuleList([FakeMedusaHead(100, [10, 20, 30, 40])])
    hidden = torch.randn(1, 1, 1)
    tree = build_candidate_tree(heads, hidden, top_k=4, max_depth=1)
    assert len(tree.tokens) == 4
    assert set(tree.tokens.tolist()) == {10, 20, 30, 40}
    assert all(p == -1 for p in tree.parents)


def test_tree_size_two_heads():
    """2 heads with top-k=3 → 3 + 9 = 12 candidates."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30]),
        FakeMedusaHead(100, [11, 21, 31]),
    ])
    hidden = torch.randn(1, 1, 1)
    tree = build_candidate_tree(heads, hidden, top_k=3, max_depth=2)
    assert len(tree.tokens) == 12  # 3 + 3*3
    # First 3 nodes are roots
    assert tree.parents[:3] == [-1, -1, -1]
    # Remaining 9 have parents in [0, 1, 2]
    for p in tree.parents[3:]:
        assert p in [0, 1, 2]


def test_tree_paths_count():
    """2 heads with top-k=3 → 9 root-to-leaf paths."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30]),
        FakeMedusaHead(100, [11, 21, 31]),
    ])
    hidden = torch.randn(1, 1, 1)
    tree = build_candidate_tree(heads, hidden, top_k=3, max_depth=2)
    paths = get_tree_paths(tree)
    assert len(paths) == 9  # 3 * 3


def test_tree_mask_self_attention():
    """Every node should attend to itself."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20]),
        FakeMedusaHead(100, [11, 21]),
    ])
    hidden = torch.randn(1, 1, 1)
    tree = build_candidate_tree(heads, hidden, top_k=2, max_depth=2)
    mask = candidate_tree_to_mask(tree)
    n = len(tree.parents)
    for i in range(n):
        assert mask[i, i].item(), f"Node {i} should attend to itself"


def test_tree_mask_ancestor_chain():
    """Children should attend to their parents."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20]),
        FakeMedusaHead(100, [11, 21]),
    ])
    hidden = torch.randn(1, 1, 1)
    tree = build_candidate_tree(heads, hidden, top_k=2, max_depth=2)
    mask = candidate_tree_to_mask(tree)
    for i, parent in enumerate(tree.parents):
        if parent != -1:
            assert mask[i, parent].item(), (
                f"Node {i} should attend to parent {parent}"
            )


def test_tree_mask_no_cross_branch():
    """Nodes on different branches should NOT attend to each other."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20]),
        FakeMedusaHead(100, [11, 21]),
    ])
    hidden = torch.randn(1, 1, 1)
    tree = build_candidate_tree(heads, hidden, top_k=2, max_depth=2)
    mask = candidate_tree_to_mask(tree)
    # Node 0 and node 1 are roots → they shouldn't attend to each other
    assert not mask[0, 1].item(), "Root 0 should not attend to root 1"
    assert not mask[1, 0].item(), "Root 1 should not attend to root 0"


# ── Acceptance tests ─────────────────────────────────────────────────


def test_accept_all():
    """All draft tokens match → accept all."""
    draft = [10, 20, 30]
    # Create logits where argmax at each position = draft token
    logits = torch.zeros(3, 100)
    logits[0, 10] = 10.0
    logits[1, 20] = 10.0
    logits[2, 30] = 10.0
    accepted, rejected_at = accept_draft_tokens(draft, logits)
    assert accepted == [10, 20, 30]
    assert rejected_at == 3


def test_accept_none():
    """First token mismatches → accept nothing."""
    draft = [10, 20, 30]
    logits = torch.zeros(3, 100)
    logits[0, 99] = 10.0  # Wrong
    accepted, rejected_at = accept_draft_tokens(draft, logits)
    assert accepted == []
    assert rejected_at == 0


def test_accept_partial():
    """Partial acceptance stops at first mismatch."""
    draft = [10, 20, 30]
    logits = torch.zeros(3, 100)
    logits[0, 10] = 10.0
    logits[1, 20] = 10.0
    logits[2, 99] = 10.0  # Mismatch at index 2
    accepted, rejected_at = accept_draft_tokens(draft, logits)
    assert accepted == [10, 20]
    assert rejected_at == 2


# ── Path extraction tests ───────────────────────────────────────────


def test_paths_cover_all_leaves():
    """Every leaf should appear in exactly one path."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20, 30]),
        FakeMedusaHead(100, [11, 21, 31]),
    ])
    hidden = torch.randn(1, 1, 1)
    tree = build_candidate_tree(heads, hidden, top_k=3, max_depth=2)
    paths = get_tree_paths(tree)

    # Find leaves (nodes with no children)
    children_set = set()
    for i, p in enumerate(tree.parents):
        if p != -1:
            children_set.add(p)
    leaves = {i for i in range(len(tree.parents)) if i not in children_set}

    path_leaves = {path[-1] for path in paths}
    assert path_leaves == leaves, "All leaves must be covered by paths"


def test_path_monotonic_depth():
    """Each path should go strictly deeper (parent → child)."""
    heads = nn.ModuleList([
        FakeMedusaHead(100, [10, 20]),
        FakeMedusaHead(100, [11, 21]),
        FakeMedusaHead(100, [12, 22]),
    ])
    hidden = torch.randn(1, 1, 1)
    tree = build_candidate_tree(heads, hidden, top_k=2, max_depth=3)
    paths = get_tree_paths(tree)

    for path in paths:
        for i in range(1, len(path)):
            child = path[i]
            parent = path[i - 1]
            assert tree.parents[child] == parent, (
                f"In path, node {child}'s parent should be {parent}, "
                f"got {tree.parents[child]}"
            )
