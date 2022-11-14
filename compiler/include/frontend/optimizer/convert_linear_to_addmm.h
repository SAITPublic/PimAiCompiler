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

#include "common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  Convert aten::linear to aten::addmm, to enable PIM acceleartion and custom GEMV computation.
 *
 *             |
 *          prim::If
 *        /          \                                |
 *   aten::addmm   aten::matmul                       |
 *       |            |             ----->       aten::addmm
 *       |         aten::add                          |
 *        \          /                                |
 *         prim::EndIf
 *              |
 *
 *  structure of aten::linear
 *
 **/
class ConvertLinearToAddmm : public Pass
{
   public:
    ConvertLinearToAddmm();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~ConvertLinearToAddmm() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> linear_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
