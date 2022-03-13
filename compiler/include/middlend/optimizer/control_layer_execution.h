#pragma once

#include "compiler/include/frontend/optimizer/pass.h"

#include "new_ir/include/nn_model.h"
#include "new_ir/include/nn_network.h"
#include "new_ir/include/types.h"

namespace nn_compiler
{

namespace middlend
{

class ControlLayerExecution : public Pass
{
   public:
    ControlLayerExecution();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~ControlLayerExecution() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> control_layers_;

};  // class ControlLayerExecution

}  // namespace middlend
}  // namespace nn_compiler
