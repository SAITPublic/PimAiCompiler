#pragma once

#include "compiler/include/frontend/optimizer/pass.h"

#include "new_ir/include/nn_model.h"
#include "new_ir/include/nn_network.h"

#include "new_ir/include/types.h"

namespace nn_compiler
{
namespace frontend
{
class CatLabeling : public Pass
{
   public:
    CatLabeling();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~CatLabeling() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> cat_labeling_layers_;
};  // class CatLabeling

}  // namespace frontend
}  // namespace nn_compiler
