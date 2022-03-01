#pragma once

#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/passes/pass_support.hpp"
#include "compiler/include/middlend/utils/op_basic_util.hpp"

#include "new_ir/include/nn_network.h"
#include "new_ir/include/types.h"
namespace nn_compiler
{

namespace frontend
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

}  // namespace frontend
}  // namespace nn_compiler
