#pragma once

#ifndef NNCOMPILER_PRIM_OP_H
#define NNCOMPILER_PRIM_OP_H

#include <torch/script.h>
#include <unordered_map>
#include <vector>

#include "new_runtime/include/common/log.hpp"
#include "new_runtime/include/executor/prim_utils.h"

namespace nn_compiler
{
namespace runtime
{
torch::Tensor primData(const torch::Tensor& input_tensor);

c10::Device primDevice(const torch::Tensor& input_tensor);

int64_t primDtype(const torch::Tensor& input_tensor);

template <typename T>
T& primEndIf(T& inputs) { return inputs; }

std::vector<torch::Tensor> primEndLoop(const std::vector<torch::Tensor>& inputs);

bool primIf(bool cond);

void primListConstruct(std::vector<torch::IValue>& stack, size_t num_inputs, at::ListTypePtr type);

void primListConstruct(std::vector<torch::IValue>& stack);

void primListUnpack(std::vector<torch::IValue>& stack, size_t num_outputs);

void primLoop(int max_trip_cnt, torch::Tensor& cond, std::unordered_map<int, torch::Tensor>& blobs);

void primRaiseException(std::string msg);

// Provide 3 kinds Constant
// template <typename T>
// T primScalarConstant(T* data_ptr);

template <typename T>
T primScalarConstant(T* data_ptr)
{
    if (std::is_same<T, int64_t>::value || std::is_same<T, int32_t>::value || std::is_same<T, float>::value ||
        std::is_same<T, double>::value) {
        return *data_ptr;
    } else {
        Log::RT::E() << "Unsupported scalar type!";
    }
}

// STRING, DEVICE
std::string primStrConstsnt(void* data_ptr);

void primTupleConstruct(std::vector<torch::IValue>& stack, size_t num_inputs);

torch::IValue primTupleIndex(const std::vector<torch::IValue>& inputs, int64_t index);

std::vector<torch::IValue> primTupleUnpack(c10::intrusive_ptr<c10::ivalue::Tuple> tuple);

template <typename T>
T& primUncheckedCast(T& inputs) { return inputs; }

torch::IValue primUninitialized();

at::IValue primVariable(std::string ntype, std::vector<torch::IValue> inputs);

}  // namespace runtime
}  // namespace nn_compiler

#endif  // NNCOMPILER_PRIM_OP_H
