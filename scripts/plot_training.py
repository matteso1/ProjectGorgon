"""Plot training loss curve from training_log.json.

Reads the training log produced by the Medusa head training pipeline
and generates a loss curve PNG for the README / blog post.

Usage:
    python scripts/plot_training.py [--input training_log.json] [--output reports/loss_curve.png]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training loss curve")
    parser.add_argument("--input", default="training_log.json",
                        help="Path to training_log.json")
    parser.add_argument("--output", default="reports/loss_curve.png",
                        help="Output PNG path")
    parser.add_argument("--title", default="Medusa Head Training Loss",
                        help="Plot title")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate log format without plotting")
    return parser.parse_args(argv)


def load_training_log(path: str | Path) -> tuple[list[int], list[float], dict | None]:
    """Load training log, supporting multiple formats.

    Returns (steps, losses, config_or_None).

    Supported formats:
    - Dict with "loss_history" (bare float list) and optional "config"
    - Dict with "steps" list of {step, loss} dicts
    - JSON array of {step, loss} dicts
    - JSONL with {step, loss} per line
    """
    path = Path(path)
    text = path.read_text()

    steps: list[int] = []
    losses: list[float] = []
    config = None

    # Try JSON first
    try:
        data = json.loads(text)

        # Format: {"config": {...}, "loss_history": [float, ...]}
        if isinstance(data, dict) and "loss_history" in data:
            raw_losses = data["loss_history"]
            config = data.get("config")
            total_steps = config.get("max_steps", len(raw_losses)) if config else len(raw_losses)
            # Distribute steps evenly across loss entries
            n = len(raw_losses)
            steps = [int(i * total_steps / n) for i in range(n)]
            losses = [float(l) for l in raw_losses]
            return steps, losses, config

        # Format: {"steps": [{step, loss}, ...]}
        if isinstance(data, dict) and "steps" in data:
            entries = data["steps"]
        elif isinstance(data, list):
            entries = data
        else:
            entries = [data]

        for entry in entries:
            step = entry.get("step", entry.get("global_step"))
            loss = entry.get("loss", entry.get("train_loss"))
            if step is not None and loss is not None:
                steps.append(int(step))
                losses.append(float(loss))
        return steps, losses, config

    except json.JSONDecodeError:
        pass

    # Fall back to JSONL
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            entry = json.loads(line)
            step = entry.get("step", entry.get("global_step"))
            loss = entry.get("loss", entry.get("train_loss"))
            if step is not None and loss is not None:
                steps.append(int(step))
                losses.append(float(loss))
    return steps, losses, config


def main() -> None:
    args = parse_args()

    log_path = Path(args.input)
    if not log_path.exists():
        print(f"Training log not found: {log_path}")
        print("Run training first or provide --input path.")
        sys.exit(1)

    steps, losses, config = load_training_log(log_path)
    print(f"Loaded {len(losses)} loss entries from {log_path}")
    if config:
        print(f"Model: {config.get('model_name', 'unknown')}")
        print(f"Heads: {config.get('num_heads', '?')}, LR: {config.get('learning_rate', '?')}")

    if not steps:
        print("No valid step/loss entries found in log.")
        sys.exit(1)

    print(f"Steps: {min(steps)} to {max(steps)}")
    print(f"Loss: {max(losses):.4f} -> {min(losses):.4f}")

    if args.dry_run:
        print("Dry-run: skipping plot generation.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, losses, linewidth=1.5, color="#2563eb")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(args.title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add annotations
    min_loss_idx = losses.index(min(losses))
    ax.annotate(
        f"Min: {losses[min_loss_idx]:.4f}",
        xy=(steps[min_loss_idx], losses[min_loss_idx]),
        xytext=(steps[min_loss_idx], losses[min_loss_idx] + 0.1),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
        color="gray",
    )

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
