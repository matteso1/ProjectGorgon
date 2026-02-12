"""Kernel implementations for Project Gorgon."""

try:
    from gorgon.kernels.fused_tree_verify_triton import fused_tree_verify
except ImportError:
    fused_tree_verify = None
