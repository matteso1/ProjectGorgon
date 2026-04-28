#!/bin/bash
# ================================================================
#  Project Gorgon -- RunPod Training Script (one-shot)
#
#  Usage: Run on a RunPod A100 pod with runpod/pytorch:2.4.0 image
#
#    export HF_TOKEN=hf_YOUR_TOKEN_HERE
#    git clone -b gorgon-impl https://github.com/matteso1/ProjectGorgon.git
#    cd ProjectGorgon
#    bash scripts/setup_and_train.sh
#
# ================================================================
set -euo pipefail

echo "============================================================"
echo " Project Gorgon -- RunPod Setup & Train"
echo "============================================================"
echo ""

# -- 1. Verify GPU ---------------------------------------------------
echo "[1/8] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected! Make sure you're on a GPU pod."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# -- 2. Check PyTorch version ----------------------------------------
echo "[2/8] Checking PyTorch..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "   PyTorch: $TORCH_VERSION"

if [[ "$TORCH_VERSION" == "none" ]]; then
    echo "ERROR: PyTorch not found! Use a PyTorch template on RunPod."
    exit 1
fi

TORCH_MINOR=$(python -c "import torch; v=torch.__version__.split('.'); print(v[1])")
if [[ "$TORCH_MINOR" -lt 5 ]]; then
    echo "   -> PyTorch < 2.5 detected, pinning transformers==4.45.2"
    TRANSFORMERS_PIN="transformers==4.45.2"
else
    echo "   -> PyTorch >= 2.5, using latest transformers"
    TRANSFORMERS_PIN="transformers"
fi

# -- 3. Install dependencies -----------------------------------------
echo ""
echo "[3/8] Installing dependencies..."
pip install --no-cache-dir -q \
    "$TRANSFORMERS_PIN" \
    "bitsandbytes>=0.43.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.30.0" \
    "tqdm" \
    "pyyaml"

echo "   Done"
echo ""

# -- 4. Verify imports -----------------------------------------------
echo "[4/8] Verifying imports..."
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
echo ""

# -- 5. Check HF token -----------------------------------------------
echo "[5/8] Checking HF token..."
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "   HF_TOKEN not set!"
    read -rp "   Enter your HF token (or press Enter to abort): " HF_TOKEN
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: HF_TOKEN is required for Llama-3"
        exit 1
    fi
    export HF_TOKEN
fi
echo "   HF_TOKEN is set"
echo ""

# -- 6. Install project ----------------------------------------------
echo "[6/8] Installing project..."
pip install --no-cache-dir -q -e . 2>/dev/null || true
echo "   Done"
echo ""

# -- 7. Train ---------------------------------------------------------
echo "[7/8] Starting training..."
echo "   Config: batch_size=4, grad_accum=4, warmup=200, cosine LR"
echo "   Steps: 10000 (~2.5 hrs on A100)"
echo "   Checkpoints: ./checkpoints/ (saving every 2000 steps, keeping 1)"
echo ""

python scripts/train_medusa_heads.py \
    --config configs/train_heads.yaml

echo ""

# -- 8. Run benchmark -------------------------------------------------
echo "[8/8] Running benchmark..."
echo "   Comparing speculative vs baseline generation"
echo ""

# Check if benchmark script exists and run it
if [[ -f scripts/benchmark_inference.py ]]; then
    python scripts/benchmark_inference.py \
        --num-trials 5 \
        --warmup-steps 2 \
        --max-new-tokens 128 \
        2>&1 || echo "   Benchmark had errors (non-fatal)"
else
    echo "   Benchmark script not found, skipping"
fi

echo ""
echo "============================================================"
echo " DONE!"
echo ""
echo " Checkpoints: ./checkpoints/"
echo " Training log: ./checkpoints/training_log.json"
echo " Benchmark:    ./reports/ (if available)"
echo ""
echo " NEXT STEPS:"
echo "   1. Download checkpoints/  and reports/"
echo "   2. Stop this pod to save money!"
echo ""
echo " To download from your local machine:"
echo "   runpodctl receive <pod-id>:checkpoints/"
echo "   -- or use the RunPod web UI file manager --"
echo "============================================================"
