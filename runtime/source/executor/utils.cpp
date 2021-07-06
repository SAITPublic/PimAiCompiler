#include <string>
#include <torch/script.h>

namespace nnrt
{

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor) { return torch::jit::IValue(tensor); }

torch::jit::IValue strToIValue(std::string str) { return torch::jit::IValue(str); }


}  // namespace nnrt
