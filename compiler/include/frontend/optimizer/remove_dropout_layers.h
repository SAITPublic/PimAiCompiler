#pragma once

#include "compiler/include/common/pass.hpp"

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
