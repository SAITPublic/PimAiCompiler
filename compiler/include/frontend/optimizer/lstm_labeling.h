#pragma once

#include "compiler/include/frontend/optimizer/pass.h"

#include "new_ir/include/nn_model.h"
#include "new_ir/include/nn_network.h"

#include "new_ir/include/types.h"

namespace nn_compiler
{

namespace frontend
{

class LstmLabeling : public Pass
{
   public:
    LstmLabeling();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~LstmLabeling() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> aten_lstm1_layers_;
};  // class LstmLabeling

}  // namespace frontend
}  // namespace nn_compiler
