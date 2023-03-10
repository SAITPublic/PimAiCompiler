/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#pragma once

#ifndef NNCOMPILER_PRIM_OP_H
#define NNCOMPILER_PRIM_OP_H

#include <torch/script.h>

#include "executor/utils/utils.h"

namespace nn_compiler
{
namespace runtime
{
namespace op_executor
{
torch::Tensor primData(const torch::Tensor& input_tensor);

c10::Device primDevice(const torch::Tensor& input_tensor);

int64_t primDtype(const torch::Tensor& input_tensor);

template <typename T>
T& primEndIf(T& inputs)
{
    return inputs;
}

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
        DLOG(FATAL) << "Unsupported scalar type!";
    }
}

// STRING, DEVICE
std::string primStrConstsnt(void* data_ptr);

void primTupleConstruct(std::vector<torch::IValue>& stack, size_t num_inputs);

torch::IValue primTupleIndex(const std::vector<torch::IValue>& inputs, int64_t index);

std::vector<torch::IValue> primTupleUnpack(c10::intrusive_ptr<c10::ivalue::Tuple> tuple);

template <typename T>
T& primUncheckedCast(T& inputs)
{
    return inputs;
}

torch::IValue primUninitialized();

at::IValue primVariable(std::string ntype, std::vector<torch::IValue> inputs);

}  // namespace op_executor
}  // namespace runtime
}  // namespace nn_compiler

#endif  // NNCOMPILER_PRIM_OP_H
