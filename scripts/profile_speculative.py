"""Torch profiler wrapper for speculative decoding.

Wraps speculative_generate with torch.profiler.profile, exports a
Chrome trace for visualization in chrome://tracing, and prints a
summary table of the top 20 ops by CUDA time.

Usage:
    python scripts/profile_speculative.py --model-name <model>
    python scripts/profile_speculative.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

PROMPT = (
    "Explain the concept of speculative decoding in large language models "
    "and how tree-structured verification improves throughput."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile speculative decoding")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="reports/speculative_trace.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print usage info without running")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.dry_run:
        print("Profiler dry-run mode -- no GPU required.")
        print(f"Would profile: {args.model_name}")
        print(f"Output trace: {args.output}")
        print("Open trace with: chrome://tracing")
        return

    import torch
    from torch.profiler import profile, ProfilerActivity

    from gorgon.inference.gorgon_loop import speculative_generate
    from gorgon.models.backbone import load_backbone_4bit

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_name}")
    model, tokenizer, heads = load_backbone_4bit(
        model_name=args.model_name,
        num_heads=4,
        device_map=device,
    )

    # Warmup
    print("Warmup run...")
    speculative_generate(
        model=model,
        tokenizer=tokenizer,
        heads=heads,
        prompt=PROMPT,
        max_new_tokens=16,
        top_k=args.top_k,
        device=device,
    )

    # Profiled run
    print("Profiling...")
    activities = [ProfilerActivity.CPU]
    if device != "cpu":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        result = speculative_generate(
            model=model,
            tokenizer=tokenizer,
            heads=heads,
            prompt=PROMPT,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            device=device,
        )

    # Export Chrome trace
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(output_path))
    print(f"Chrome trace saved to: {output_path}")
    print("Open in browser: chrome://tracing")

    # Print summary
    sort_key = "cuda_time_total" if device != "cpu" else "cpu_time_total"
    print(f"\nTop 20 ops by {sort_key}:")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    print(f"\nGenerated {len(result.generated_ids)} tokens")
    print(f"Acceptance rate: {result.acceptance_rate:.2%}")
    if result.iteration_stats:
        print(f"Mean accepted length (tau): {result.mean_accepted_length:.2f}")


if __name__ == "__main__":
    main()
