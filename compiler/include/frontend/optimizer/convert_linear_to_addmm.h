#pragma once

#include "compiler/include/common/pass.hpp"
#include "ir/include/layers/all_layers.h"

namespace nn_compiler
{
namespace frontend
{
class ConvertLinearToAddmm : public Pass
{
   public:
    ConvertLinearToAddmm();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~ConvertLinearToAddmm() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> linear_layers_;
};

}  // namespace frontend
}  // namespace nn_compiler
