# Benchmark Results

## System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

## Small run (n=64, d=64, iters=10, warmup=2)
- ref_ms: 0.090
- triton_ms: 0.032
- cuda_ms: 0.043
- speedup triton vs ref: 2.794×
- speedup cuda vs ref: 2.080×

## Larger run (n=256, d=64, iters=50, warmup=5)
- ref_ms: 0.104
- triton_ms: 0.035
- cuda_ms: 0.170
- speedup triton vs ref: 2.999×
- speedup cuda vs ref: 0.608×

## Notes
- Triton kernel currently outperforms the naive CUDA kernel.
- CUDA kernel is correct but not optimized yet; planned next step is tiling + better memory access.

## Phase 5: End-to-end Llama-3-8B (dry-run)

### System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

### Config
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Max new tokens: 128
- Prompt max length: 256
- Trials: 5
- Warmup: 2
- Seed: 0
- Medusa heads: 4
- Top-k: 4

### Baseline (autoregressive)
- Tokens: 100
- Elapsed (s): 10.00
- Tokens/s: 10.00
- Acceptance rate: n/a

### Speculative (Medusa)
- Tokens: 100
- Elapsed (s): 5.00
- Tokens/s: 20.00
- Acceptance rate: 0.50

### Speedup
- Speculative vs baseline: 2.00x

## Phase 5: End-to-end Llama-3-8B (TODO)

### System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

### Config
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Max new tokens: 16
- Prompt max length: 64
- Trials: 1
- Warmup: 1
- Seed: 0
- Medusa heads: 4
- Top-k: 4

### Baseline (autoregressive)
- Tokens: 64
- Elapsed (s): 3.20
- Tokens/s: 19.98
- Acceptance rate: n/a

### Speculative (Medusa)
- Tokens: 0
- Elapsed (s): 2.40
- Tokens/s: 0.00
- Acceptance rate: 0.00

### Speedup
- Speculative vs baseline: 0.00x

## Phase 5: End-to-end Llama-3-8B (2026-02-01)

### System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

### Config
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Max new tokens: 16
- Prompt max length: 64
- Trials: 1
- Warmup: 1
- Seed: 0
- Medusa heads: 4
- Top-k: 4

### Baseline (autoregressive)
- Tokens: 64
- Elapsed (s): 2.53
- Tokens/s: 25.26
- Acceptance rate: n/a

### Speculative (Medusa)
- Tokens: 0
- Elapsed (s): 3.01
- Tokens/s: 0.00
- Acceptance rate: 0.00

### Speedup
- Speculative vs baseline: 0.00x

## Phase 5: End-to-end Llama-3-8B (2026-02-01)

### System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

### Config
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Max new tokens: 8
- Prompt max length: 64
- Trials: 1
- Warmup: 1
- Seed: 0
- Medusa heads: 1
- Top-k: 4

### Baseline (autoregressive)
- Tokens: 32
- Elapsed (s): 2.39
- Tokens/s: 13.40
- Acceptance rate: n/a

### Speculative (Medusa)
- Tokens: 0
- Elapsed (s): 0.64
- Tokens/s: 0.00
- Acceptance rate: 0.00

### Speedup
- Speculative vs baseline: 0.00x

## Phase 5: End-to-end Llama-3-8B (2026-02-01)

### System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

### Config
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Max new tokens: 8
- Prompt max length: 64
- Trials: 1
- Warmup: 1
- Seed: 0
- Medusa heads: 1
- Top-k: 4

### Baseline (autoregressive)
- Tokens: 32
- Elapsed (s): 2.17
- Tokens/s: 14.75
- Acceptance rate: n/a

### Speculative (Medusa)
- Tokens: 0
- Elapsed (s): 1.09
- Tokens/s: 0.00
- Acceptance rate: 0.00

### Speedup
- Speculative vs baseline: 0.00x

## Phase 5: End-to-end Llama-3-8B (2026-02-01)

### System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

### Config
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Max new tokens: 8
- Prompt max length: 64
- Trials: 1
- Warmup: 0
- Seed: 0
- Medusa heads: 1
- Top-k: 4

### Baseline (autoregressive)
- Tokens: 32
- Elapsed (s): 6.88
- Tokens/s: 4.65
- Acceptance rate: n/a

### Speculative (Medusa)
- Tokens: 0
- Elapsed (s): 1.32
- Tokens/s: 0.00
- Acceptance rate: 0.00

### Speedup
- Speculative vs baseline: 0.00x

## Phase 5: End-to-end Llama-3-8B (2026-02-01)

### System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

### Config
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Max new tokens: 8
- Prompt max length: 64
- Trials: 1
- Warmup: 0
- Seed: 0
- Medusa heads: 1
- Top-k: 4

### Baseline (autoregressive)
- Tokens: 32
- Elapsed (s): 2.94
- Tokens/s: 10.87
- Acceptance rate: n/a

### Speculative (Medusa)
- Tokens: 0
- Elapsed (s): 1.13
- Tokens/s: 0.00
- Acceptance rate: 0.00

### Speedup
- Speculative vs baseline: 0.00x

## Phase 5: End-to-end Llama-3-8B (2026-02-01)

### System
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Torch: 2.10.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU

### Config
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- Max new tokens: 8
- Prompt max length: 64
- Trials: 1
- Warmup: 0
- Seed: 0
- Medusa heads: 1
- Top-k: 4

### Baseline (autoregressive)
- Tokens: 32
- Elapsed (s): 9.62
- Tokens/s: 3.33
- Acceptance rate: n/a

### Speculative (Medusa)
- Tokens: 0
- Elapsed (s): 1.96
- Tokens/s: 0.00
- Acceptance rate: 0.00

### Speedup
- Speculative vs baseline: 0.00x