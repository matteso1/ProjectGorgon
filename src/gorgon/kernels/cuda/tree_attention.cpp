#include <torch/extension.h>
#include <vector>
#include <algorithm>

// CUDA forward declaration
std::vector<torch::Tensor> tree_attention_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask);

std::vector<torch::Tensor> tree_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask) {

  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");

  TORCH_CHECK(q.dim() == 2, "q must be 2D");
  TORCH_CHECK(k.dim() == 2, "k must be 2D");
  TORCH_CHECK(v.dim() == 2, "v must be 2D");
  TORCH_CHECK(mask.dim() == 2, "mask must be 2D");

  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");

  TORCH_CHECK(q.scalar_type() == torch::kFloat, "q must be float32");
  TORCH_CHECK(k.scalar_type() == torch::kFloat, "k must be float32");
  TORCH_CHECK(v.scalar_type() == torch::kFloat, "v must be float32");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");

  auto n = q.size(0);
  auto d = q.size(1);

  TORCH_CHECK(k.size(0) == n && k.size(1) == d, "k must be [n, d]");
  TORCH_CHECK(v.size(0) == n && v.size(1) == d, "v must be [n, d]");
  TORCH_CHECK(mask.size(0) == n && mask.size(1) == n, "mask must be [n, n]");

  return tree_attention_cuda_forward(q, k, v, mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tree_attention_forward, "Tree Attention forward (CUDA)");
}