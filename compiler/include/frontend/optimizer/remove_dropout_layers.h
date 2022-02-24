#pragma once

#include "compiler/include/frontend/optimizer/pass.h"
#include "new_ir/include/layers/aten_dropout_layer.h"

namespace nn_compiler
{
namespace frontend
{

class RemoveDropoutLayers : public Pass
{
   public:
    RemoveDropoutLayers();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveDropoutLayers() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> remove_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
