#pragma once

#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/passes/pass_support.hpp"
#include "compiler/include/middlend/utils/op_basic_util.hpp"

#include "new_ir/include/nn_network.h"

namespace nn_compiler
{

namespace frontend
{

class UpdateLayerId : public Pass
{
   public:
    UpdateLayerId();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~UpdateLayerId() = default;

   private:
};  // class UpdateLayerId

}  // namespace frontend
}  // namespace nn_compiler
