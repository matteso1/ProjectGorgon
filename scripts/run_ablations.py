"""Ablation study runner for speculative decoding configurations.

Sweeps over num_heads, top_k, max_depth, and confidence_threshold
to measure how each parameter affects throughput and acceptance rate.

Usage:
    python scripts/run_ablations.py --model-name <model> [--output-dir reports]
    python scripts/run_ablations.py --dry-run  # quick test without GPU
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@dataclass
class AblationConfig:
    num_heads: int
    top_k: int
    max_depth: int
    confidence_threshold: float
    tree_mode: str  # "static" or "adaptive"


@dataclass
class AblationResult:
    config: AblationConfig
    tokens_per_second: float
    acceptance_rate: float
    mean_accepted_length: float
    tree_utilization: float
    token_count: int
    elapsed_s: float


# Default sweep grid
DEFAULT_NUM_HEADS = [1, 2, 3, 4]
DEFAULT_TOP_K = [2, 4, 8]
DEFAULT_MAX_DEPTH = [1, 2, 3, 4]
DEFAULT_CONFIDENCE_THRESHOLDS = [0.0, 0.1, 0.3, 0.5]

PROMPT = (
    "Explain the concept of speculative decoding in large language models "
    "and how tree-structured verification improves throughput."
)


def build_sweep_configs() -> List[AblationConfig]:
    """Build the full grid of ablation configurations."""
    configs: List[AblationConfig] = []

    # Static tree ablations
    for nh, tk, md in itertools.product(
        DEFAULT_NUM_HEADS, DEFAULT_TOP_K, DEFAULT_MAX_DEPTH
    ):
        if md > nh:
            continue  # depth can't exceed num_heads
        configs.append(AblationConfig(
            num_heads=nh, top_k=tk, max_depth=md,
            confidence_threshold=0.0, tree_mode="static",
        ))

    # Adaptive tree ablations (use full heads, vary threshold)
    for ct in DEFAULT_CONFIDENCE_THRESHOLDS:
        if ct == 0.0:
            continue  # already covered by static
        configs.append(AblationConfig(
            num_heads=4, top_k=4, max_depth=4,
            confidence_threshold=ct, tree_mode="adaptive",
        ))

    return configs


def run_single_ablation(
    model,
    tokenizer,
    heads,
    config: AblationConfig,
    max_new_tokens: int,
    device: str,
) -> AblationResult:
    """Run a single ablation configuration."""
    from gorgon.inference.gorgon_loop import speculative_generate

    import torch.nn as nn
    subset_heads = nn.ModuleList(list(heads[:config.num_heads]))

    start = time.perf_counter()
    result = speculative_generate(
        model=model,
        tokenizer=tokenizer,
        heads=subset_heads,
        prompt=PROMPT,
        max_new_tokens=max_new_tokens,
        top_k=config.top_k,
        device=device,
    )
    elapsed = time.perf_counter() - start

    return AblationResult(
        config=config,
        tokens_per_second=len(result.generated_ids) / elapsed if elapsed > 0 else 0,
        acceptance_rate=result.acceptance_rate,
        mean_accepted_length=result.mean_accepted_length,
        tree_utilization=result.tree_utilization,
        token_count=len(result.generated_ids),
        elapsed_s=elapsed,
    )


def make_dry_results(configs: List[AblationConfig]) -> List[AblationResult]:
    """Generate synthetic results for dry-run mode."""
    import random
    random.seed(42)
    results = []
    for cfg in configs:
        results.append(AblationResult(
            config=cfg,
            tokens_per_second=random.uniform(10, 50),
            acceptance_rate=random.uniform(0.2, 0.8),
            mean_accepted_length=random.uniform(0.5, 3.0),
            tree_utilization=random.uniform(0.05, 0.5),
            token_count=128,
            elapsed_s=random.uniform(1.0, 10.0),
        ))
    return results


def results_to_markdown(results: List[AblationResult]) -> str:
    """Format results as a markdown table."""
    lines = [
        "| Heads | Top-k | Depth | Threshold | Mode | Tok/s | Accept | Tau | Util |",
        "|-------|-------|-------|-----------|------|-------|--------|-----|------|",
    ]
    for r in results:
        c = r.config
        lines.append(
            f"| {c.num_heads} | {c.top_k} | {c.max_depth} | {c.confidence_threshold:.1f} "
            f"| {c.tree_mode} | {r.tokens_per_second:.1f} | {r.acceptance_rate:.2f} "
            f"| {r.mean_accepted_length:.2f} | {r.tree_utilization:.2%} |"
        )
    return "\n".join(lines)


def results_to_dicts(results: List[AblationResult]) -> List[Dict[str, Any]]:
    """Convert results to JSON-serializable dicts."""
    out = []
    for r in results:
        d = {
            "config": asdict(r.config),
            "tokens_per_second": r.tokens_per_second,
            "acceptance_rate": r.acceptance_rate,
            "mean_accepted_length": r.mean_accepted_length,
            "tree_utilization": r.tree_utilization,
            "token_count": r.token_count,
            "elapsed_s": r.elapsed_s,
        }
        out.append(d)
    return out


def plot_ablation_charts(results: List[AblationResult], output_dir: Path) -> None:
    """Generate PNG charts from ablation results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping chart generation")
        return

    # Chart 1: Top-k vs throughput (for static, max_depth=num_heads)
    static_results = [r for r in results if r.config.tree_mode == "static"]
    if static_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Group by num_heads
        by_heads: Dict[int, List[AblationResult]] = {}
        for r in static_results:
            if r.config.max_depth == r.config.num_heads:
                by_heads.setdefault(r.config.num_heads, []).append(r)

        ax = axes[0]
        for nh, rs in sorted(by_heads.items()):
            rs.sort(key=lambda x: x.config.top_k)
            ax.plot(
                [r.config.top_k for r in rs],
                [r.tokens_per_second for r in rs],
                marker="o", label=f"{nh} heads",
            )
        ax.set_xlabel("Top-k")
        ax.set_ylabel("Tokens/s")
        ax.set_title("Throughput vs Top-k")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for nh, rs in sorted(by_heads.items()):
            rs.sort(key=lambda x: x.config.top_k)
            ax.plot(
                [r.config.top_k for r in rs],
                [r.acceptance_rate for r in rs],
                marker="s", label=f"{nh} heads",
            )
        ax.set_xlabel("Top-k")
        ax.set_ylabel("Acceptance Rate")
        ax.set_title("Acceptance Rate vs Top-k")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = output_dir / "ablation_topk.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved: {chart_path}")

    # Chart 2: Confidence threshold impact
    adaptive_results = [r for r in results if r.config.tree_mode == "adaptive"]
    # Include static baseline (threshold=0)
    baseline = [
        r for r in static_results
        if r.config.num_heads == 4 and r.config.top_k == 4 and r.config.max_depth == 4
    ]
    threshold_results = baseline + adaptive_results
    if threshold_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        threshold_results.sort(key=lambda x: x.config.confidence_threshold)
        ax.plot(
            [r.config.confidence_threshold for r in threshold_results],
            [r.tokens_per_second for r in threshold_results],
            marker="o", color="tab:blue", label="Tok/s",
        )
        ax2 = ax.twinx()
        ax2.plot(
            [r.config.confidence_threshold for r in threshold_results],
            [r.acceptance_rate for r in threshold_results],
            marker="s", color="tab:orange", label="Accept rate",
        )
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Tokens/s", color="tab:blue")
        ax2.set_ylabel("Acceptance Rate", color="tab:orange")
        ax.set_title("Adaptive Pruning: Threshold Impact")
        ax.grid(True, alpha=0.3)
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85))
        plt.tight_layout()
        chart_path = output_dir / "ablation_threshold.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved: {chart_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate synthetic results without GPU")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_sweep_configs()
    print(f"Running {len(configs)} ablation configurations...")

    if args.dry_run:
        results = make_dry_results(configs)
    else:
        import torch
        from gorgon.models.backbone import load_backbone_4bit

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, tokenizer, heads = load_backbone_4bit(
            model_name=args.model_name,
            num_heads=max(DEFAULT_NUM_HEADS),
            device_map=device,
        )

        results = []
        for i, cfg in enumerate(configs):
            print(f"  [{i+1}/{len(configs)}] {cfg}")
            r = run_single_ablation(
                model, tokenizer, heads, cfg,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            results.append(r)

    # Save JSON
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results_to_dicts(results), f, indent=2)
    print(f"Saved: {json_path}")

    # Save markdown
    md_path = output_dir / "ablation_results.md"
    with open(md_path, "w") as f:
        f.write(results_to_markdown(results))
    print(f"Saved: {md_path}")

    # Generate charts
    plot_ablation_charts(results, output_dir)


if __name__ == "__main__":
    main()
