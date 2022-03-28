#pragma once

#include "compiler/include/common/pass.hpp"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  If inputs of a prim::ListConstrcut are all constants, it is possible to do the work of prim::ListConstruct
 *  directly in frontend. The constructed value will be stored in a prim::Variable layer.
 **/
class ConstructList : public Pass
{
   public:
    ConstructList();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& graph_model);

    ~ConstructList() = default;

   private:
    std::vector<
        std::pair<std::shared_ptr<nn_compiler::ir::NNLayer>, std::vector<std::shared_ptr<nn_compiler::ir::DTensor>>>>
        process_layer_and_dtensor_;
};
}  // namespace frontend
}  // namespace nn_compiler
