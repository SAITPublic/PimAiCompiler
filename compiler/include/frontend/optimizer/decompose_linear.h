#pragma once

#include "compiler/include/common/pass.hpp"
#include "ir/include/layers/all_layers.h"

namespace nn_compiler
{
namespace frontend
{
class DecomposeLinear : public Pass
{
   public:
    DecomposeLinear();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~DecomposeLinear() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> linear_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
