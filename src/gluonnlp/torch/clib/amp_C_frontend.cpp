#include <torch/extension.h>

void multi_tensor_lans_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int bias_correction,
  const float weight_decay,
  const int grad_averaging,
  const int mode,
  const bool normalize_grad);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_lans", &multi_tensor_lans_cuda,
        "Computes and apply update for LANS optimizer");
}
