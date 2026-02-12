# Speculative Decoding Fix + Retrain Design

**Date**: 2026-02-11
**Status**: Ready for implementation (retrain on RunPod)

## Problem

The A100 benchmark showed 0.2% acceptance rate and 0.08x "speedup" (12x slower than baseline). Three root causes identified:

### 1. Benchmark never loads trained checkpoints (SHOWSTOPPER)

`benchmark_inference.py` calls `load_backbone_4bit()` which creates fresh `MedusaHead` instances with randomly initialized `ResidualBlock`. The 10k-step training produced checkpoints, but the benchmark has no `--heads-checkpoint` flag. The benchmark was measuring random heads.

### 2. Missing RMSNorm in MedusaHead (ARCHITECTURAL)

Llama-3 pipeline: `hidden -> RMSNorm -> lm_head -> logits`.
Our code extracts `outputs.hidden_states[-1]` (pre-norm) and feeds it to the head's `lm_head` (initialized from backbone, expects post-norm input). The single `ResidualBlock` must learn to approximate RMSNorm AND predict shifted tokens simultaneously.

### 3. Training config issues (MINOR)

Cosine scheduler decayed LR to 0.0 (no `eta_min`). The resumed run from step 6500 was in the tail of cosine decay with near-zero LR. Effective training was mostly in the first run.

## Solution

### Architecture: MedusaHead with frozen RMSNorm

```
Input: hidden_states (pre-norm)
  -> [RMSNorm] (frozen copy of backbone's model.model.norm)
  -> [ResidualBlock] (Linear + SiLU + skip connection)
  -> [lm_head] (initialized from backbone)
  -> logits
```

At initialization with untrained ResidualBlock, each head computes approximately `lm_head(RMSNorm(hidden))` -- matching the backbone's own next-token prediction. Training only needs to learn the delta for shifted-position prediction.

### Benchmark: checkpoint loading

Added `--heads-checkpoint` CLI flag to `benchmark_inference.py`. Uses new `load_trained_heads()` utility from `backbone.py` that handles old/new checkpoint format mismatch gracefully.

### Training: updated config

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| max_steps | 10,000 | 30,000 | Converge properly |
| learning_rate | 1e-4 | 3e-4 | Easier optimization with norm |
| warmup_steps | 50/200 | 500 | Stable higher LR |
| cosine eta_min | 0 | 1e-5 | Never flatline at zero LR |
| optimizer | AdamW(all params) | AdamW(trainable only) | Don't track frozen norm state |

## Files Changed

| File | Change |
|------|--------|
| `src/gorgon/models/medusa_heads.py` | Added optional `norm` parameter to `MedusaHead` |
| `src/gorgon/models/backbone.py` | `_get_backbone_norm()`, norm copying in `load_backbone_4bit()`, `load_trained_heads()` |
| `src/gorgon/benchmarks/cli.py` | `--heads-checkpoint` CLI flag |
| `scripts/benchmark_inference.py` | Checkpoint loading before benchmark |
| `scripts/train_medusa_heads.py` | `eta_min=1e-5`, trainable-only optimizer + grad clipping |
| `configs/train_heads.yaml` | 30k steps, 3e-4 LR, 500 warmup |

## Retrain Command

```bash
python scripts/train_medusa_heads.py \
  --config configs/train_heads.yaml \
  --checkpoint-dir checkpoints
```

## Benchmark Command (after training)

```bash
python scripts/benchmark_inference.py \
  --heads-checkpoint checkpoints/medusa_heads_latest.pt \
  --num-trials 5 --warmup-steps 2 --max-new-tokens 128
```

## Expected Outcome

With proper norm initialization and sufficient training:
- Head 0 (next token): should approach backbone accuracy quickly
- Heads 1-3 (tokens +2 to +4): lower but meaningful acceptance rates
- Target: 40-60% overall acceptance rate, 1.5-2.5x speedup
