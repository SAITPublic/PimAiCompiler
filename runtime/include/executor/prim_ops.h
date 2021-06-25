#pragma once

#include <vector>
#include <torch/script.h>
#include "../nnrt_types.h"

namespace nnrt
{
__NNRT_API__ c10::Device primDevice(const torch::Tensor& input_tensor);

__NNRT_API__ int64_t primDtype(const torch::Tensor& input_tensor);

__NNRT_API__ torch::Tensor primData(const torch::Tensor& input_tensor);

__NNRT_API__ torch::IValue primUninitialized();

__NNRT_API__ template <typename T>
T primUncheckedCast(const T& inputs);

__NNRT_API__ void primRaiseException(std::string& msg);

__NNRT_API__ torch::Tensor primTupleIndex(const std::vector<torch::Tensor>& inputs, int64_t index);

__NNRT_API__ void primTupleConstruct(std::vector<torch::IValue>& stack, size_t num_inputs);

__NNRT_API__ void primTupleUnpack(std::vector<torch::IValue>& stack);

__NNRT_API__ void primListConstruct(std::vector<torch::IValue>& stack, size_t num_inputs, at::ListTypePtr type);

__NNRT_API__ void primListUnpack(std::vector<torch::IValue>& stack, size_t num_outputs);

// Provide 3 kinds Constant
__NNRT_API__ template <typename T>
T primScalarConstant(T* data_ptr);

// STRING, DEVICE
__NNRT_API__ std::string primStrConstsnt(void* data_ptr);

__NNRT_API__ torch::Tensor primTensorConstant(void* data_ptr, std::vector<int64_t>& shape, DataType dtype);

}  // namespace nnrt
