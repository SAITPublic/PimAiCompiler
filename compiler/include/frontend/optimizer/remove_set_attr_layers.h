#pragma once

#include "compiler/include/common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
/***@Detail:
 *  There is no need to keep prim::SetAttr in our graph, it always comes out afer a prim::Variabl layer,
 *  and update its value by a computation result.
 *  the connection for prim::SetAttr is always like:
 *
 *       prim::Variable    a computation layer (e.g. aten::ge)
 *          |       \        /
 *          |      prim::SetAttr
 *          |
 *     prim::GetAttr
 *          |
 *
 *  The prim::Variable which is updated by prim::SetAttr is never used before, so we can remove
 *  prim::SetAttr and prim::Variable directly, change the structure to:
 *
 *         a computation layer (e.g. aten::ge)
 *                  |
 *            prim::GetAttr
 *                  |
 *
 *  Some prim::GetAttr may get attr from a computation layer rather than prim::Variable, so we
 *  remove them in later pass together.
 *
 ***/

class RemoveSetAttrLayers : public Pass
{
   public:
    RemoveSetAttrLayers();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveSetAttrLayers() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> remove_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
