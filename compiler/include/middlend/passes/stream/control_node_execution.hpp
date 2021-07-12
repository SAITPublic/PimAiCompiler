#pragma once

#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/passes/pass_support.hpp"
#include "compiler/include/middlend/utils/op_basic_util.hpp"
#include "ir/include/all_nodes.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/node.hpp"

namespace nn_compiler {

class ControlNodeExecutionPass : public PassMixin<ControlNodeExecutionPass> {
 public:
    RetVal initialize(const UtilManager& util_manager, const TraitManager& trait_manager);

    RetVal run(nn_ir::NNIR& graph, CompilationContext& context);
}; // class ControlNodeExecutionPass

} // namespace nn_compiler
