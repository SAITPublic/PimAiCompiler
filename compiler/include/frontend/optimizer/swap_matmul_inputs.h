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

#include "common/pass.hpp"
#include "frontend/optimizer/utils/constant_parser.h"
#include "half.hpp"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  1. Change structure 1 to structure 2 to run custom GEMV in runtime.
 *                                                                                |
 *      |                                tansposed weight (prim::Constant)   aten::transpose
 *      |     weight (prim::Constant)                                \           /
 *      \       /                                                     aten::matmul
 *     aten::matmul                                                       |
 *         |                                                        aten::transpose
 *         |    /                                                          \         /
 *       aten::add                                                          aten::add
 *           |                                                                  |
 **/
class SwapMatmulInputs : public Pass
{
   public:
    SwapMatmulInputs();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~SwapMatmulInputs() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;

    optimizer_utils::ConstantParser constant_parser_;
};

}  // namespace frontend
}  // namespace nn_compiler
