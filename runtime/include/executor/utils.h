#pragma once

#include <torch/script.h>

namespace nnrt
{
template <typename T>
torch::jit::IValue scalarToIValue(const T& scalar)
{
    return torch::jit::IValue(scalar);
}

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor);

torch::jit::IValue strToIValue(std::string str);

}  // namespace nnrt
