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

#include <assert.h>
#include <torch/script.h>
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler
{
namespace frontend
{
class AttrParser
{
   public:
    AttrParser() = default;

    int64_t getIntAttr(const torch::jit::Node* node, c10::Symbol symbol);

    std::vector<int64_t> getIntArrayAttr(const torch::jit::Node* node, c10::Symbol symbol);

    double getFP64Attr(const torch::jit::Node* node, c10::Symbol symbol);

    std::string getStrAttr(const torch::jit::Node* node, c10::Symbol symbol);

    at::Tensor getTensorAttr(const torch::jit::Node* node, c10::Symbol symbol);

    std::pair<std::vector<at::Tensor>, std::vector<at::Tensor> > getGeneralWeightAndBias(const torch::jit::Node* node,
                                                                                         int weight_idx, int bias_idx);

    std::pair<std::vector<at::Tensor>, std::vector<at::Tensor> > getLstmWeightAndBias(const torch::jit::Node* node);

    ~AttrParser() = default;
};

std::shared_ptr<nn_compiler::ir::DTensor> getDTensorData(const torch::jit::Node* node_constant);

void ptTensor2DTensor(at::Tensor torch_tensor, std::shared_ptr<nn_compiler::ir::DTensor> d_tensor);

}  // namespace frontend
}  // namespace nn_compiler
