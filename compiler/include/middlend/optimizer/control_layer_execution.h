#pragma once

#include "compiler/include/common/pass.hpp"

#include "ir/include/nn_model.h"
#include "ir/include/nn_network.h"
#include "ir/include/types.h"

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
