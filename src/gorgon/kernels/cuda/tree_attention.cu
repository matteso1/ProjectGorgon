#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

__global__ void tree_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    const bool* mask,
    float* out,
    int n,
    int d) {

  int row = blockIdx.x;
  int col = threadIdx.x;

  if (row >= n) return;

  extern __shared__ float scores[];

  // compute scores for this row
  if (col < n) {
    float acc = 0.0f;
    for (int i = 0; i < d; i++) {
      acc += q[row * d + i] * k[col * d + i];
    }
    float scale = rsqrtf((float)d);
    float s = acc * scale;
    scores[col] = mask[row * n + col] ? s : -INFINITY;
  }
  __syncthreads();

  // softmax normalization
  float max_s = -INFINITY;
  for (int j = 0; j < n; j++) max_s = fmaxf(max_s, scores[j]);

  float denom = 0.0f;
  for (int j = 0; j < n; j++) denom += expf(scores[j] - max_s);

  // output for this column
  if (col < d) {
    float out_val = 0.0f;
    for (int j = 0; j < n; j++) {
      float w = expf(scores[j] - max_s) / denom;
      out_val += w * v[j * d + col];
    }
    out[row * d + col] = out_val;
  }
}

std::vector<torch::Tensor> tree_attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask) {

  const auto n = q.size(0);
  const auto d = q.size(1);

  auto out = torch::zeros_like(q);

  // threads >= max(n, d), but <= 1024
  int needed = (int)std::max(n, d);
  int threads = 1;
  while (threads < needed) threads <<= 1;
  if (threads > 1024) {
    throw std::runtime_error("n and d must be <= 1024 for this naive CUDA kernel");
  }

  tree_attention_kernel<<<(int)n, threads, (size_t)n * sizeof(float)>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      mask.data_ptr<bool>(),
      out.data_ptr<float>(),
      (int)n,
      (int)d
  );

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }

  return {out};
}