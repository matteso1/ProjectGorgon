from __future__ import annotations

import argparse
import random
import time
import platform

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


import torch

from gorgon.kernels.tree_mask import build_tree_mask
from gorgon.kernels.tree_attention_ref import tree_attention_ref
from gorgon.kernels.tree_attention_triton import tree_attention_triton

try:
    from gorgon.kernels.tree_attention_cuda import tree_attention_cuda
    _CUDA_EXT_ERROR = None
except Exception as exc:
    tree_attention_cuda = None
    _CUDA_EXT_ERROR = exc


def make_random_tree(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    parents = [-1]
    for i in range(1, n):
        parents.append(rng.randrange(0, i))
    return parents


def time_fn(fn, iters: int, device: str, *args) -> float:
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=== System Info ===")
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    parents = make_random_tree(args.n, args.seed)
    mask = build_tree_mask(parents).to(device)

    q = torch.randn(args.n, args.d, device=device)
    k = torch.randn(args.n, args.d, device=device)
    v = torch.randn(args.n, args.d, device=device)

    # warmup
    for _ in range(args.warmup):
        tree_attention_ref(q, k, v, mask)
        if device == "cuda":
            tree_attention_triton(q, k, v, mask)
            if tree_attention_cuda is not None:
                tree_attention_cuda(q, k, v, mask)

    results = {}

    results["ref_ms"] = time_fn(tree_attention_ref, args.iters, device, q, k, v, mask)

    if device == "cuda":
        results["triton_ms"] = time_fn(tree_attention_triton, args.iters, device, q, k, v, mask)
        if tree_attention_cuda is not None:
            results["cuda_ms"] = time_fn(tree_attention_cuda, args.iters, device, q, k, v, mask)
        else:
            print("CUDA extension unavailable:", _CUDA_EXT_ERROR)

    print("=== Results (ms per call) ===")
    for k_name, v_ms in results.items():
        print(f"{k_name}: {v_ms:.3f}")

    if "triton_ms" in results:
        print("speedup triton vs ref:", results["ref_ms"] / results["triton_ms"])
    if "cuda_ms" in results:
        print("speedup cuda vs ref:", results["ref_ms"] / results["cuda_ms"])


if __name__ == "__main__":
    main()