#pragma once

#include "compiler/include/common/pass.hpp"
#include "new_ir/include/layers/prim_constant_layer.h"
#include "new_ir/include/layers/prim_variable_layer.h"

namespace nn_compiler
{

namespace frontend
{

class RemoveConstantLayers : public Pass
{
   public:
    RemoveConstantLayers();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~RemoveConstantLayers() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> remove_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
