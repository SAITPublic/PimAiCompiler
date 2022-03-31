#pragma once

#include "compiler/include/common/pass.hpp"
#include "ir/include/layers/all_layers.h"

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
