#pragma once

#include "compiler/include/common/pass.hpp"
#include "compiler/include/frontend/optimizer/utils/constant_parser.h"

#include "half.hpp"
#include "ir/include/nn_network.h"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  1. Change structure 1 to structure 2 to run custom GEMV in runtime.
 *                                                                               |
 *          |                                tansposed weight(prim::Constant)  aten::transpose
 *   bias   |     weight (prim::Constant)                     bias     |       /
 *      \       /                                                \     |      /
 *     aten::addmm                                                aten::addmm
 *          |                                                          |
 *                                                              aten::transpose
 *                                                                     |
 *
 *     structure 1                                              sturcture 2
 **/

class SwapAddmmInputs : public Pass
{
   public:
    SwapAddmmInputs();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~SwapAddmmInputs() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;

    ConstantParser constant_parser_;
};

}  // namespace frontend
}  // namespace nn_compiler
