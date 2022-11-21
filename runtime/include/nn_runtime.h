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

#include "builder/model_builder.h"
#include "executor/stream_executor.h"
#include "ir/include/nn_model.h"

namespace nn_compiler
{
namespace runtime
{
class NNRuntime
{
   public:
    NNRuntime() {}

    NNRuntime(std::unique_ptr<nn_compiler::ir::NNModel>& model, std::string model_type = "");

    void inferenceModel(const std::vector<torch::Tensor>& input_tensors, std::vector<torch::Tensor>& output_tensors,
                        bool profiling = false);

    int rocblas_init(void);

    int test(void);

    ~NNRuntime();

   private:
    std::shared_ptr<ModelBuilder> mbuilder_ = nullptr;

    std::shared_ptr<StreamExecutor> executor_ = nullptr;

    std::string model_type_ = "";
};

}  // namespace runtime
}  // namespace nn_compiler
