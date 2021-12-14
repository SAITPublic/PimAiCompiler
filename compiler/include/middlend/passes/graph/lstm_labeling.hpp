#pragma once

#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/passes/pass_support.hpp"
#include "compiler/include/middlend/utils/op_basic_util.hpp"
#include "ir/include/all_nodes.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/node.hpp"

/***
 * @ detail:
 * LSTM pattern recognition for connection:
 *               |
 *          aten::lstm1
 *               |
 *     prim::tuple_construct
 *               |
 *         aten::append
 *               |
 * 
 * Find and set the matched aten::lstm1 for runtime optimization.
 **/

namespace nn_compiler {

class LstmLabelingPass : public PassMixin<LstmLabelingPass> {
 public:
    RetVal initialize(const UtilManager& util_manager, const TraitManager& trait_manager);

    RetVal run(nn_ir::NNIR& graph, CompilationContext& context);
}; // class LstmLabelingPass

} // namespace nn_compiler
