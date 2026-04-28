#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  Project Gorgon — Cloud GPU Training Guide
# ════════════════════════════════════════════════════════════════
#
#  Memory requirements for full Medusa training:
#    Head params (bf16):   ~4.3 GB
#    Adam states (fp32):  ~17.3 GB
#    Backbone (4-bit):     ~5.0 GB
#    Activations:          ~3-5 GB
#    ─────────────────────────────
#    TOTAL:               ~30-32 GB
#
#  ┌──────────────┬──────────┬────────────┬─────────────┐
#  │ Provider     │ GPU      │ VRAM       │ ~Cost/hr    │
#  ├──────────────┼──────────┼────────────┼─────────────┤
#  │ RunPod       │ A100 40G │ 40 GB      │ ~$1.50      │
#  │ RunPod       │ A100 80G │ 80 GB      │ ~$2.00      │
#  │ Colab Pro+   │ A100     │ 40 GB      │ ~$50/mo     │
#  │ GCP          │ A100     │ 40 GB      │ ~$3.67      │
#  │ Lambda Labs  │ A100     │ 40 GB      │ ~$1.10      │
#  └──────────────┴──────────┴────────────┴─────────────┘
#
#  Estimated training time: ~20-40 min for 500 steps on A100
#
# ════════════════════════════════════════════════════════════════

# ── OPTION 1: RunPod (recommended — cheapest + easiest) ──────
#
# 1. Go to https://www.runpod.io
# 2. Create an account, add $5-10 credit
# 3. Deploy a GPU Pod:
#      Template: "RunPod PyTorch 2.1" (or any CUDA 12.x image)
#      GPU:      A100 40GB (or 80GB)
#      Disk:     50 GB
# 4. SSH in or use the web terminal, then run:

# --- Copy this block into your RunPod terminal ---
set -e

# Clone the repo
git clone https://github.com/YOUR_USERNAME/ProjectGorgon.git
cd ProjectGorgon

# Set up environment
pip install torch transformers bitsandbytes datasets accelerate tqdm pyyaml

# Set your Hugging Face token
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Run training (bf16 on A100)
python scripts/train_medusa_heads.py \
    --config configs/train_heads.yaml \
    --steps 500 \
    --head-dtype bf16 \
    --save-every 50

# Download checkpoints when done
# (use RunPod's file browser or scp)
# Checkpoints will be in ./checkpoints/


# ── OPTION 2: Google Colab Pro+ ─────────────────────────────
#
# 1. Open a Colab notebook
# 2. Runtime → Change runtime type → A100
# 3. Run these cells:
#
# Cell 1:
#   !git clone https://github.com/YOUR_USERNAME/ProjectGorgon.git
#   %cd ProjectGorgon
#   !pip install torch transformers bitsandbytes datasets accelerate tqdm pyyaml
#
# Cell 2:
#   import os
#   os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
#
# Cell 3:
#   !python scripts/train_medusa_heads.py \
#       --config configs/train_heads.yaml \
#       --steps 500 --head-dtype bf16 --save-every 50
#
# Cell 4 (download checkpoints):
#   from google.colab import files
#   !tar czf checkpoints.tar.gz checkpoints/
#   files.download("checkpoints.tar.gz")


# ── OPTION 3: GCP (you have $50 credit) ─────────────────────
#
# 1. Create a VM with GPU:
#   gcloud compute instances create gorgon-train \
#     --zone=us-central1-a \
#     --machine-type=a2-highgpu-1g \
#     --accelerator=type=nvidia-tesla-a100,count=1 \
#     --boot-disk-size=100GB \
#     --image-family=pytorch-latest-gpu \
#     --image-project=deeplearning-platform-release
#
# 2. SSH in:
#   gcloud compute ssh gorgon-train --zone=us-central1-a
#
# 3. Run the same training commands as Option 1 above.
#
# 4. IMPORTANT: Stop the instance when done!
#   gcloud compute instances stop gorgon-train --zone=us-central1-a


# ── After training: copy checkpoints back ────────────────────
#
# From your local machine:
#   scp -r runpod:/workspace/ProjectGorgon/checkpoints/ \
#       /path/to/ProjectGorgon/.worktrees/gorgon-impl/checkpoints/
#
# Then run the benchmark locally:
#   python scripts/benchmark_inference.py \
#       --num-trials 3 --max-new-tokens 64
