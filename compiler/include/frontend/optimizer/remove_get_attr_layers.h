#pragma once

#include "compiler/include/common/pass.hpp"

namespace nn_compiler {

namespace frontend {

/***@Detail:
 *  There is no need to keep prim::GetAttr in our graph, because the connection for prim::GetAttr
 *  is always like:
 *       prim::Variable (might be deleted by: remove_set_attr_layers)
 *            |
 *       prim::GetAttr
 *            |
 *    a computation op (e.g. aten::len)
 *            |
 *
 *  We can remove prim::GetAttr directly, change to:
 *      prim::Variable (might be deleted)
 *            |
 *    a computation op (e.g. aten::len)
 *            |
***/

class RemoveGetAttrLayers : public Pass {
 public:
    RemoveGetAttrLayers();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveGetAttrLayers() = default;

 private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> remove_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
