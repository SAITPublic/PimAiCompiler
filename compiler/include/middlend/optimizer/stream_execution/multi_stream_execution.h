#pragma once

#include "common/pass.hpp"
#include "ir/include/utils/graph_util.h"

namespace nn_compiler
{
namespace middlend
{
/** @Details:
 *  Automatically recognize parallelizable model structure, and apply multi-stream to acclerate execution time.
 *  For Switch Transformer model, the following modle structure is recognized:
 *                                aten::contiguous
 *                      /          |           |            \
 *               aten::slice  aten::slice  aten::slice  aten::slice
 *                     |           |           |             |
 *         aten::transpose aten::transpose aten::transpose aten::transpose
 *                     |           |           |             |
 *             aten::matmul  aten::matmul  aten::matmul   aten::matmul
 *                     |           |           |             |
 *               aten::add     aten::add    aten::add      aten::add
 *                     |           |           |             |
 *           prim::Variable prim::Variable prim::Variable prim::Variable
 *                     |           |           |             |
 *           aten::reshape  aten::reshape  aten::reshape   aten::reshape
 *                     \           |           |             /
 *                              prim::ListConstruct
 *  And then get reorgnized by a new multi-stream Op, which will execute four aten::matmul Ops inside with multi-stream
 *  in runtime.
 **/
class MultiStreamExecution : public Pass
{
   public:
    MultiStreamExecution();

    bool fitCondition(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    void run(std::unique_ptr<nn_compiler::ir::NNModel>& model);

    ~MultiStreamExecution() = default;

   private:
    bool backwardToCheckAndFindStartLayer(
        std::unique_ptr<nn_compiler::ir::NNModel>& model,
        std::shared_ptr<nn_compiler::ir::NNLayer>& multi_stream_start_layer,
        std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>& multi_stream_apply_layers);

    void process(std::unique_ptr<nn_compiler::ir::NNModel>& model, int start_idx);

    std::map<std::shared_ptr<nn_compiler::ir::NNLayer>, std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>>
        start_to_stream_layers_;

    std::map<std::shared_ptr<nn_compiler::ir::NNLayer>, std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>>
        pre_layers_map_;

    std::vector<std::vector<std::shared_ptr<nn_compiler::ir::NNLayer>>> branches_;

    int max_search_level_ = 10;

};  // class MultiStreamExecution

}  // namespace middlend
}  // namespace nn_compiler
