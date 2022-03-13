#pragma once

#include "compiler/include/common/pass.hpp"
#include "compiler/include/frontend/optimizer/utils/constant_parser.h"

#include "half.hpp"
#include "ir/include/nn_network.h"

namespace nn_compiler {

namespace frontend {

/** @Details:
 *  1. Change structure 1 to structure 2 to run custom GEMV in runtime.
 *                                                                                |
 *      |                                tansposed weight (prim::Constant)   aten::transpose
 *      |     weight (prim::Constant)                                \           /
 *      \       /                                                     aten::matmul
 *     aten::matmul                                                       |
 *         |                                                        aten::transpose
 *         |    /                                                          \         /
 *       aten::add                                                          aten::add
 *           |                                                                  |
 **/

class SwapMatmulInputs : public Pass {
 public:
    SwapMatmulInputs();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~SwapMatmulInputs() = default;

 private:
    std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>> layers_;

    ConstantParser constant_parser_;

    void transpose_for_constant(std::shared_ptr<nn_compiler::ir::NNLayer>& layer);
};

}  // namespace frontend
}  // namespace nn_compiler
