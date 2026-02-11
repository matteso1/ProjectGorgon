#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  Project Gorgon — RunPod Training Script (one-shot)
#
#  Usage: Run on a RunPod A100 pod with runpod/pytorch:2.4.0 image
#
#    git clone -b gorgon-impl https://github.com/matteso1/ProjectGorgon.git
#    cd ProjectGorgon
#    bash scripts/setup_and_train.sh
#
# ════════════════════════════════════════════════════════════════
set -euo pipefail

echo "============================================================"
echo " Project Gorgon — RunPod Setup & Train"
echo "============================================================"
echo ""

# ── 1. Verify GPU ────────────────────────────────────────────────
echo "🔍 Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "❌ No GPU detected! Make sure you're on a GPU pod."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 2. Check PyTorch version and install compatible deps ─────────
echo "🔍 Checking PyTorch version..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "   PyTorch: $TORCH_VERSION"

if [[ "$TORCH_VERSION" == "none" ]]; then
    echo "❌ PyTorch not found! Use a PyTorch template on RunPod."
    exit 1
fi

# PyTorch 2.4.x doesn't have set_submodule, so we need transformers<=4.45.2
# PyTorch 2.5+ has set_submodule, so any transformers version works
TORCH_MINOR=$(python -c "import torch; v=torch.__version__.split('.'); print(v[1])")
echo "   PyTorch minor version: $TORCH_MINOR"

if [[ "$TORCH_MINOR" -lt 5 ]]; then
    echo "   → PyTorch < 2.5 detected, pinning transformers==4.45.2"
    TRANSFORMERS_PIN="transformers==4.45.2"
else
    echo "   → PyTorch >= 2.5 detected, using latest transformers"
    TRANSFORMERS_PIN="transformers"
fi

# ── 3. Install dependencies ─────────────────────────────────────
echo ""
echo "📦 Installing dependencies..."
pip install --no-cache-dir -q \
    "$TRANSFORMERS_PIN" \
    "bitsandbytes>=0.43.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.30.0" \
    "tqdm" \
    "pyyaml"

echo "   ✓ All dependencies installed"
echo ""

# ── 4. Verify imports work ───────────────────────────────────────
echo "🔍 Verifying imports..."
python -c "
import torch
import transformers
import bitsandbytes
import datasets
import accelerate
print(f'   torch:        {torch.__version__}')
print(f'   transformers: {transformers.__version__}')
print(f'   bitsandbytes: {bitsandbytes.__version__}')
print(f'   datasets:     {datasets.__version__}')
print(f'   accelerate:   {accelerate.__version__}')
print(f'   CUDA:         {torch.cuda.is_available()}')
print(f'   GPU:          {torch.cuda.get_device_name(0)}')
print(f'   VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
echo "   ✓ All imports verified"
echo ""

# ── 5. Check HF token ───────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "⚠  HF_TOKEN not set! Set it before running:"
    echo "   export HF_TOKEN=hf_YOUR_TOKEN_HERE"
    echo ""
    read -rp "Enter your HF token (or press Enter to abort): " HF_TOKEN
    if [[ -z "$HF_TOKEN" ]]; then
        echo "❌ Aborting — HF_TOKEN is required for Llama-3"
        exit 1
    fi
    export HF_TOKEN
fi
echo "   ✓ HF_TOKEN is set"
echo ""

# ── 6. Install the project in development mode ──────────────────
echo "📦 Installing project..."
pip install --no-cache-dir -q -e . 2>/dev/null || true
echo "   ✓ Project installed"
echo ""

# ── 7. Train ─────────────────────────────────────────────────────
echo "🚀 Starting training..."
echo "   This should take ~20-30 minutes on A100."
echo "   Checkpoints saved to ./checkpoints/"
echo ""

python scripts/train_medusa_heads.py \
    --config configs/train_heads.yaml \
    --steps 500 \
    --save-every 50

echo ""
echo "============================================================"
echo " ✅ Training complete!"
echo " Checkpoints are in: ./checkpoints/"
echo " Download them and stop this pod!"
echo "============================================================"
