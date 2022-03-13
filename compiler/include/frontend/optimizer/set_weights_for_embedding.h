#pragma once

#include "compiler/include/common/pass.hpp"
#include "compiler/include/frontend/optimizer/utils/constant_parser.h"

#include "half.hpp"
#include "ir/include/layers//aten_embedding_layer.h"
#include "ir/include/layers//prim_constant_layer.h"

namespace nn_compiler {

namespace frontend {

class SetWeightsForEmbedding : public Pass {
 public:
    SetWeightsForEmbedding();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~SetWeightsForEmbedding() = default;

 private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;

    ConstantParser constant_parser_;
};

}  // namespace frontend
}  // namespace nn_compiler
