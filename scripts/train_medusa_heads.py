#!/usr/bin/env python3
"""Train Medusa heads on a text dataset.

Usage:
    python scripts/train_medusa_heads.py --config configs/train_heads.yaml
    python scripts/train_medusa_heads.py --steps 500 --lr 1e-4 --dataset wikitext
    python scripts/train_medusa_heads.py --steps 100 --dataset wikitext --save-every 50

Trains lightweight Medusa draft heads on top of a frozen Llama-3-8B backbone.
The heads learn to predict future tokens, enabling speculative decoding.

Memory requirements
-------------------
Head architecture: ResidualBlock(4096) -> Linear(4096, 128256)
Per-head params: ~541M  x  4 heads  =  ~2.16B params total

With bf16 mixed-precision + Adam:
  - Head params (bf16):        ~4.3 GB
  - Adam states (fp32):       ~17.3 GB
  - Backbone (4-bit):          ~5.0 GB
  - Activations/gradients:     ~3-5 GB
  - TOTAL:                    ~30-32 GB

Recommended GPU: A100 40GB, A6000 48GB, or A100 80GB.
Colab Pro+ (A100) works. RunPod A100 40GB (~$1.50/hr) works.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import torch
import yaml
from tqdm import tqdm

from gorgon.models.backbone import load_backbone_4bit
from gorgon.models.medusa_heads import MedusaHead
from gorgon.data.dataset import get_dataloader


# --- YAML config ---------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Medusa heads")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--model-name", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(ROOT / "checkpoints"))
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--head-dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Dtype for head params (bf16 recommended)")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Linear warmup steps for LR scheduler")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI args (CLI takes precedence)."""
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    config.setdefault("model_name", args.model_name)
    config.setdefault("num_heads", args.num_heads)
    config.setdefault("max_steps", args.steps)
    config.setdefault("learning_rate", args.lr)
    config.setdefault("dataset", args.dataset)
    config.setdefault("seq_length", args.seq_length)

    config["save_every"] = args.save_every
    config["checkpoint_dir"] = args.checkpoint_dir
    config["resume"] = args.resume
    config["device"] = args.device
    config["max_samples"] = args.max_samples
    config["grad_accum"] = args.grad_accum
    config["head_dtype"] = args.head_dtype
    config["batch_size"] = args.batch_size
    config["warmup_steps"] = args.warmup_steps
    config["max_grad_norm"] = args.max_grad_norm

    # Ensure numeric types (YAML can parse 1e-4 as string in some versions)
    config["learning_rate"] = float(config["learning_rate"])
    config["max_grad_norm"] = float(config["max_grad_norm"])
    config["num_heads"] = int(config["num_heads"])
    config["max_steps"] = int(config["max_steps"])
    config["seq_length"] = int(config.get("seq_length", 512))
    config["batch_size"] = int(config["batch_size"])
    config["grad_accum"] = int(config["grad_accum"])
    config["warmup_steps"] = int(config["warmup_steps"])
    config["save_every"] = int(config["save_every"])

    return config


# --- Checkpointing -------------------------------------------------------


def save_checkpoint(
    heads: torch.nn.ModuleList,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    config: dict,
    checkpoint_dir: str,
    metrics: dict | None = None,
    heads_only: bool = False,
) -> Path:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "loss": loss,
        "heads_state_dict": heads.state_dict(),
        "config": config,
    }
    if not heads_only:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
    if metrics:
        payload["metrics"] = metrics

    if heads_only:
        path = ckpt_dir / f"medusa_heads_step{step:06d}.pt"
        torch.save(payload, path)
    else:
        path = ckpt_dir / "medusa_heads_latest.pt"
        torch.save(payload, path)

    return path


def _cleanup_old_checkpoints(checkpoint_dir: str, keep: int = 3) -> None:
    """Keep only the most recent `keep` step checkpoints."""
    ckpt_dir = Path(checkpoint_dir)
    step_ckpts = sorted(ckpt_dir.glob("medusa_heads_step*.pt"))
    for old in step_ckpts[:-keep]:
        old.unlink(missing_ok=True)


def load_checkpoint(
    path: str,
    heads: torch.nn.ModuleList,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    heads.load_state_dict(ckpt["heads_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  Resumed from step {ckpt['step']}, loss={ckpt.get('loss', '?'):.4f}")
    return ckpt["step"]


# --- Training core -------------------------------------------------------

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def build_shifted_targets(
    input_ids: torch.Tensor,
    num_heads: int,
    ignore_index: int = -100,
) -> list[torch.Tensor]:
    """Build shifted targets for each head directly.

    Head k predicts token at position t+k+1 given hidden state at t.
    Target for head k is input_ids shifted left by (k+1).
    """
    targets = []
    seq_len = input_ids.shape[1]
    for k in range(num_heads):
        shift = k + 1
        if shift < seq_len:
            target = torch.full_like(input_ids, ignore_index)
            target[:, :-shift] = input_ids[:, shift:]
        else:
            target = torch.full_like(input_ids, ignore_index)
        targets.append(target)
    return targets


def train_step_amp(
    backbone: torch.nn.Module,
    heads: torch.nn.ModuleList,
    input_ids: torch.Tensor,
    num_heads: int,
    scaler: torch.amp.GradScaler,
    head_dtype: torch.dtype,
    ignore_index: int = -100,
    grad_accum_scale: float = 1.0,
) -> float:
    """Single training step with AMP mixed precision.

    1. Forward backbone (frozen, no_grad) -> hidden states
    2. Forward each head on hidden states -> logits
    3. Cross-entropy loss against shifted targets
    4. Backward (no optimizer step -- caller handles accumulation)
    """
    backbone.eval()
    heads.train()

    # Backbone forward (frozen)
    with torch.no_grad():
        try:
            outputs = backbone(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        except TypeError:
            outputs = backbone(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1]  # (B, T, D)

    # Move hidden to head device/dtype
    head_param = next(heads.parameters())
    if hidden_states.device != head_param.device or hidden_states.dtype != head_dtype:
        hidden_states = hidden_states.to(device=head_param.device, dtype=head_dtype)

    # Build targets
    targets = build_shifted_targets(input_ids, num_heads, ignore_index)

    # Head forward + loss with AMP
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    with torch.amp.autocast("cuda", dtype=head_dtype):
        total_loss = torch.tensor(0.0, device=head_param.device)
        for k, head in enumerate(heads):
            logits = head(hidden_states)  # (B, T, V)
            target = targets[k].to(head_param.device)
            vocab_size = logits.size(-1)
            loss = criterion(logits.view(-1, vocab_size), target.view(-1))
            total_loss = total_loss + loss

        total_loss = total_loss / len(heads)  # Average across heads
        total_loss = total_loss / grad_accum_scale

    # Backward with scaler
    scaler.scale(total_loss).backward()

    return total_loss.detach().item() * grad_accum_scale


@torch.no_grad()
def validate(
    backbone: torch.nn.Module,
    heads: torch.nn.ModuleList,
    val_loader,
    num_heads: int,
    head_dtype: torch.dtype,
    device: str,
    ignore_index: int = -100,
) -> float:
    """Run validation and return average loss."""
    backbone.eval()
    heads.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    count = 0

    for batch in val_loader:
        input_ids = batch.to(device) if isinstance(batch, torch.Tensor) else batch
        try:
            outputs = backbone(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        except TypeError:
            outputs = backbone(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1]

        head_param = next(heads.parameters())
        if hidden_states.device != head_param.device or hidden_states.dtype != head_dtype:
            hidden_states = hidden_states.to(device=head_param.device, dtype=head_dtype)

        targets = build_shifted_targets(input_ids, num_heads, ignore_index)

        batch_loss = 0.0
        for k, head in enumerate(heads):
            logits = head(hidden_states)
            target = targets[k].to(head_param.device)
            vocab_size = logits.size(-1)
            loss = criterion(logits.view(-1, vocab_size), target.view(-1))
            batch_loss += loss.item()

        total_loss += batch_loss / len(heads)
        count += 1

    return total_loss / max(count, 1)


def main() -> None:
    args = parse_args()
    config = load_config(args)

    head_dtype = DTYPE_MAP[config["head_dtype"]]
    grad_accum = config["grad_accum"]
    batch_size = config["batch_size"]
    warmup_steps = config["warmup_steps"]
    max_grad_norm = config["max_grad_norm"]

    print("=" * 60)
    print(" Project Gorgon -- Medusa Head Training")
    print("=" * 60)
    print(f"  Model:        {config['model_name']}")
    print(f"  Heads:        {config['num_heads']}")
    print(f"  Steps:        {config['max_steps']}")
    print(f"  LR:           {config['learning_rate']}")
    print(f"  Dataset:      {config['dataset']}")
    print(f"  Seq len:      {config.get('seq_length', 512)}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Head dtype:   {config['head_dtype']}")
    print(f"  Grad accum:   {grad_accum}")
    print(f"  Warmup:       {warmup_steps}")
    print(f"  Max grad norm:{max_grad_norm}")
    print()

    # Device
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device:       {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU:          {gpu_name} ({gpu_mem:.1f} GB)")
        if gpu_mem < 30:
            print(f"  WARNING: {gpu_mem:.0f} GB may be insufficient. "
                  f"Recommend >=40 GB (A100, A6000).")

    # HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("  WARNING: HF_TOKEN not set. Gated models will fail.")

    # Load backbone + heads
    print("\nLoading backbone (4-bit)...")
    model, tokenizer, heads = load_backbone_4bit(
        model_name=config["model_name"],
        num_heads=config["num_heads"],
        token=hf_token,
        device_map=device,
    )

    # Move heads to device in the correct dtype
    head_device = device if device != "auto" else "cuda"
    heads = heads.to(device=head_device, dtype=head_dtype)

    # Try torch.compile for speed
    try:
        heads = torch.compile(heads)
        print("  torch.compile: enabled")
    except Exception:
        print("  torch.compile: not available, continuing without")

    param_count = sum(p.numel() for p in heads.parameters())
    param_gb = param_count * (2 if head_dtype != torch.float32 else 4) / (1024**3)
    print(f"  Head params:  {param_count:,} ({param_gb:.2f} GB in {config['head_dtype']})")

    # Optimizer (states always in fp32 for stability)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=(head_dtype == torch.float16))

    # LR Scheduler: linear warmup + cosine decay
    max_steps = config["max_steps"]
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / max(config["learning_rate"], 1e-10),
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_steps - warmup_steps, 1),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Resume
    start_step = 0
    if config.get("resume"):
        start_step = load_checkpoint(config["resume"], heads, optimizer, scheduler)

    # Load dataset
    print(f"\nLoading dataset: {config['dataset']}...")
    train_loader = get_dataloader(
        name=config["dataset"],
        tokenizer=tokenizer,
        seq_length=config.get("seq_length", 512),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        max_samples=config.get("max_samples"),
    )

    # Validation dataloader (small subset)
    val_loader = get_dataloader(
        name=config["dataset"],
        tokenizer=tokenizer,
        seq_length=config.get("seq_length", 512),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        max_samples=200,
    )

    # Training loop
    print(f"\nTraining for {max_steps} steps...\n")
    losses: list[float] = []
    t_start = time.perf_counter()
    data_iter = iter(train_loader)

    for step in tqdm(range(start_step, max_steps), desc="Training"):
        # Get data
        try:
            input_ids = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            input_ids = next(data_iter)

        # Move to backbone device
        try:
            backbone_device = next(model.parameters()).device
        except StopIteration:
            backbone_device = torch.device(device)
        input_ids = input_ids.to(backbone_device)

        # Train step with AMP
        loss = train_step_amp(
            backbone=model,
            heads=heads,
            input_ids=input_ids,
            num_heads=config["num_heads"],
            scaler=scaler,
            head_dtype=head_dtype,
            grad_accum_scale=grad_accum,
        )

        # Optimizer step (with gradient accumulation)
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(heads.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        losses.append(loss)

        # Logging
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            mem_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"  Step {step+1:5d} | Loss: {loss:.4f} | "
                f"Avg(10): {avg_loss:.4f} | LR: {current_lr:.2e} | Peak VRAM: {mem_gb:.1f} GB"
            )

        # Checkpoint + validation
        if (step + 1) % config["save_every"] == 0:
            # Validation
            val_loss = validate(
                backbone=model,
                heads=heads,
                val_loader=val_loader,
                num_heads=config["num_heads"],
                head_dtype=head_dtype,
                device=device,
            )
            tqdm.write(f"  Val loss: {val_loss:.4f}")

            metrics = {
                "step": step + 1,
                "loss": loss,
                "val_loss": val_loss,
                "avg_loss_10": sum(losses[-10:]) / len(losses[-10:]),
                "lr": optimizer.param_groups[0]["lr"],
                "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
            }
            # Periodic save: heads only (skip optimizer ~17GB)
            path = save_checkpoint(
                heads=heads, optimizer=optimizer, scheduler=scheduler,
                step=step + 1, loss=loss, config=config,
                checkpoint_dir=config["checkpoint_dir"],
                metrics=metrics, heads_only=True,
            )
            tqdm.write(f"  Saved checkpoint: {path}")

            # Full save for resume
            save_checkpoint(
                heads=heads, optimizer=optimizer, scheduler=scheduler,
                step=step + 1, loss=loss, config=config,
                checkpoint_dir=config["checkpoint_dir"],
                metrics=metrics, heads_only=False,
            )
            _cleanup_old_checkpoints(config["checkpoint_dir"], keep=3)

    # Final checkpoint + summary
    elapsed = time.perf_counter() - t_start
    final_metrics = {
        "total_steps": max_steps,
        "final_loss": losses[-1] if losses else 0.0,
        "avg_loss_last_50": sum(losses[-50:]) / len(losses[-50:]) if losses else 0.0,
        "training_time_s": elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
    }
    save_checkpoint(
        heads=heads, optimizer=optimizer, scheduler=scheduler,
        step=max_steps, loss=losses[-1] if losses else 0.0,
        config=config, checkpoint_dir=config["checkpoint_dir"],
        metrics=final_metrics, heads_only=False,
    )

    # Save training log
    log_path = Path(config["checkpoint_dir"]) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "config": {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                       for k, v in config.items()},
            "final_metrics": final_metrics,
            "loss_history": losses,
        }, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Total time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Final loss:     {losses[-1]:.4f}")
    print(f"  Avg last 50:    {final_metrics['avg_loss_last_50']:.4f}")
    print(f"  Peak VRAM:      {final_metrics['peak_vram_gb']:.1f} GB")
    print(f"  Checkpoints:    {config['checkpoint_dir']}")
    print(f"  Training log:   {log_path}")


if __name__ == "__main__":
    main()
