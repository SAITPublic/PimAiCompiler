#pragma once

#include <unordered_map>
#include <vector>
#include <torch/script.h>
#include "../nnrt_types.h"

namespace nnrt
{
c10::Device primDevice(const torch::Tensor& input_tensor);

int64_t primDtype(const torch::Tensor& input_tensor);

torch::Tensor primData(const torch::Tensor& input_tensor);

torch::IValue primUninitialized();

template <typename T>
T primUncheckedCast(const T& inputs);

void primRaiseException(std::string& msg);

torch::Tensor primTupleIndex(const std::vector<torch::Tensor>& inputs, int64_t index);

void primTupleConstruct(std::vector<torch::IValue>& stack, size_t num_inputs);

void primTupleUnpack(std::vector<torch::IValue>& stack);

void primListConstruct(std::vector<torch::IValue>& stack, size_t num_inputs, at::ListTypePtr type);

void primListUnpack(std::vector<torch::IValue>& stack, size_t num_outputs);

// Provide 3 kinds Constant
template <typename T>
T primScalarConstant(T* data_ptr);

// STRING, DEVICE
std::string primStrConstsnt(void* data_ptr);

torch::Tensor primTensorConstant(void* data_ptr, std::vector<int64_t>& shape, DataType dtype);

void primLoop(int max_trip_cnt, torch::Tensor& cond, std::unordered_map<int, torch::Tensor>& blobs);

std::vector<torch::Tensor> primEndLoop(const std::vector<torch::Tensor>& inputs);

}  // namespace nnrt
