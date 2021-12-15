#pragma once

#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/passes/pass_support.hpp"
#include "compiler/include/middlend/utils/op_basic_util.hpp"
#include "ir/include/all_nodes.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/node.hpp"

/***
 * @ detail:
 * Cat pattern recognition for connection:
 *               |         |
 *          aten::bmm  aten::lstm1
 *               |         |
 *          some node  some node
 *               |         |
*           aten::list_construct
 *               \         /
 *                \       /
 *                aten::Cat
 *                    |
 * From aten::bmm and aten::lstm1, search common aten::cat within 3 levels deep.
 * Find and set the matched aten::cat for runtime optimization.
 **/

namespace nn_compiler {

class CatLabelingPass : public PassMixin<CatLabelingPass> {
 public:
    RetVal initialize(const UtilManager& util_manager, const TraitManager& trait_manager);

    RetVal run(nn_ir::NNIR& graph, CompilationContext& context);
 private:
    const OpBasicUtil* op_basic_util_      = nullptr;
}; // class CatLabelingPass

} // namespace nn_compiler
