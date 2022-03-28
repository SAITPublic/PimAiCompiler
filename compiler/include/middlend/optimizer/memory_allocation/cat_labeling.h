#pragma once

#include "compiler/include/common/pass.hpp"

#include "ir/include/nn_model.h"
#include "ir/include/nn_network.h"
#include "ir/include/types.h"

namespace nn_compiler
{
namespace middlend
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

}  // namespace middlend
}  // namespace nn_compiler
