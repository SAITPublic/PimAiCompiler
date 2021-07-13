#pragma once

#include <torch/script.h>

#include "ir/include/nn_ir.hpp"

namespace nncir = nn_compiler::nn_ir;

namespace nnrt
{
template <typename T>
torch::jit::IValue scalarToIValue(const T& scalar) { return torch::jit::IValue(scalar); }

torch::jit::IValue tensorToIValue(const torch::Tensor& tensor);

torch::jit::IValue tensorListToIValue(const torch::TensorList& tensor_list);

torch::jit::IValue strToIValue(std::string str);

at::ScalarType convertDTypeToATScalarType(nncir::DataType dtype);

at::MemoryFormat getMemoryFormat(int optional_memory_format);

template <typename T>
torch::jit::IValue tupleToIValue(std::tuple<T, T> tuple) { return torch::jit::IValue(tuple); }

}  // namespace nnrt
