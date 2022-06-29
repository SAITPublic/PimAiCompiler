#pragma once

#include "common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
/*** @Details
 *  Two branches of strcuture 1 is based on the consideration of performance.
 *  But we attempts to apply and run custom addmm Op which can run faster than aten lib,
 *  so structure 1 could be simplified to structure 2 directly.
 *
 *             |
 *          prim::If
 *        /          \
 *  aten::addmm   aten::matmul                  |
 *       |            |                         |
 *       |        aten::add       --->     aten::addmm
 *        \          /                          |
 *         prim::EndIf                          |
 *             |
 *
 *          graph 1                           graph 2
 ***/
class RemoveIfWithAddmm : public Pass
{
   public:
    RemoveIfWithAddmm();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveIfWithAddmm() = default;

   private:
    std::vector<int> if_layer_idx_;

    // get layers which are only computed for deleted If branch
    void getDeleteLayers(std::unique_ptr<nn_compiler::ir::NNModel>& model,
                         std::shared_ptr<nn_compiler::ir::NNLayer> layer,
                         std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>& delete_layers);
};

}  // namespace frontend
}  // namespace nn_compiler
