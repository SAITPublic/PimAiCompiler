#pragma once

#include "common/pass.hpp"
#include "frontend/optimizer/utils/constant_parser.h"
#include "half.hpp"

namespace nn_compiler
{
namespace frontend
{
/** @Details:
 *  For RNNT model,aten::embedding is an operation which acts like "selecting a tensor with index" at inference time.
 *  So we can pre-process the constant weights of aten::embedding, reorganize the constant tensor to a vector of tensors
 *  which is ready for index-selecting. So that at runtime, this Op only needs to take input index value, and get the
 *  selected Tensor directly.
 **/
class SetWeightsForEmbedding : public Pass
{
   public:
    SetWeightsForEmbedding();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~SetWeightsForEmbedding() = default;

   private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;

    optimizer_utils::ConstantParser constant_parser_;
};

}  // namespace frontend
}  // namespace nn_compiler
