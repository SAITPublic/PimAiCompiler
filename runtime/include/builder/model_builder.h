/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#pragma once

#include <torch/script.h>

#include "common/include/types.hpp"
#include "half.hpp"
#include "ir/include/layers/all_layers.h"
#include "ir/include/nn_model.h"
#include "ir/include/tensors/data_tensor.h"
#include "ir/include/types.h"

namespace nn_compiler
{
namespace runtime
{
using namespace nn_compiler::ir;
class ModelBuilder
{
   public:
    typedef std::unordered_map<int64_t, std::pair<DataType, torch::jit::IValue>> data_store_type;

    RetVal preProcess(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    RetVal preloadModel(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    std::pair<std::vector<int64_t>, std::vector<int64_t>> loadWeightAndBias(std::vector<at::Tensor> weight_data,
                                                                            std::vector<at::Tensor> bias_data);

    data_store_type getPreLoadedData() { return preloaded_data_container_; }

   private:
    data_store_type preloaded_data_container_;

    int64_t preload_id_ = 0;
};

}  // namespace runtime
}  // namespace nn_compiler
