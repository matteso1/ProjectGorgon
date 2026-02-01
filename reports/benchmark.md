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