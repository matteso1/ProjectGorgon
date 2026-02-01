from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-3 benchmark runner")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-max-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-medusa-heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--head-train-steps", type=int, default=0)
    parser.add_argument("--head-train-lr", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(Path("reports") / "benchmark.md"),
    )
    return parser.parse_args(argv)
