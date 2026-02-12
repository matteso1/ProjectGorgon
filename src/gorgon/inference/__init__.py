"""Gorgon inference module — speculative decoding with Medusa heads."""
from gorgon.inference.gorgon_loop import (
    accept_draft_tokens,
    speculative_generate,
    baseline_generate,
    IterationStats,
    SpeculativeResult,
)
from gorgon.inference.kv_cache import GorgonKVCache, apply_cache_slice
from gorgon.inference.tree_candidates import (
    CandidateTree,
    build_candidate_tree,
    candidate_tree_to_mask,
    get_tree_paths,
)

__all__ = [
    "accept_draft_tokens",
    "speculative_generate",
    "baseline_generate",
    "IterationStats",
    "SpeculativeResult",
    "GorgonKVCache",
    "apply_cache_slice",
    "CandidateTree",
    "build_candidate_tree",
    "candidate_tree_to_mask",
    "get_tree_paths",
]
