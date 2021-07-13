#include "common/include/common.hpp"
#include "ir/include/ir_includes.hpp"
#include "ir/include/ir_types.hpp"
#include "compiler/include/common/log.hpp"

#include "compiler/include/middlend/passes/stream/control_node_execution.hpp"

#include <stack>

namespace nn_compiler {

/**
 * @brief.      initialize a DeviceLabelingPass
 * @param[in].  UtilManager& util_manager, TraitManager& trait_manager.
 * @param[out].
 * @returns.    return RetVal
 */
RetVal ControlNodeExecutionPass::initialize(const UtilManager& util_manager, const TraitManager& trait_manager) {
    return RetVal::SUCCESS;
}

/**
 * @brief.      run a ControlNodeExecutionPass
 * @details.    This function adds jump information for control nodes
 * @param[in].  nn_ir::NNIR& graph, CompilationContext& context.
 * @param[out].
 * @returns.    return RetVal
 */
RetVal ControlNodeExecutionPass::run(nn_ir::NNIR& graph, CompilationContext& context) {
    std::stack<nn_compiler::nn_ir::PrimIfNode*> if_nodes;
    std::stack<nn_compiler::nn_ir::PrimEndIfNode*> then_net_end_if_nodes;
    std::stack<nn_compiler::nn_ir::PrimLoopNode*> loop_nodes;
    bool then_net = true;
    for (auto& node : graph.getNodes()) {
        if (node.getNodeType() == nn_compiler::nn_ir::NodeType::PRIMIF) {
            auto prim_if_node = cast_if<nn_ir::PrimIfNode>(node);
            if_nodes.push(prim_if_node);
            if_nodes.push(prim_if_node);
            then_net = true;
        } else if (node.getNodeType() == nn_compiler::nn_ir::NodeType::PRIMENDIF) {
            auto prim_if_node = if_nodes.top();
            auto prim_end_if_node = cast_if<nn_ir::PrimEndIfNode>(node);
            prim_end_if_node->setIfNodeId(prim_if_node->getId());
            if (then_net) {
                prim_if_node->setElseNetStartNode(prim_end_if_node->getId() + 1);
                prim_end_if_node->setIsElseNet(false);
                then_net_end_if_nodes.push(prim_end_if_node);
            } else {
                auto then_net_end_if_node = then_net_end_if_nodes.top();
                then_net_end_if_node->setGotoNode(prim_end_if_node->getId());
                prim_end_if_node->setGotoNode(prim_end_if_node->getId() + 1);
                prim_end_if_node->setIsElseNet(true);
                then_net_end_if_nodes.pop();
            }
            if_nodes.pop();
            then_net = !then_net;
        } else if (node.getNodeType() == nn_compiler::nn_ir::NodeType::PRIMLOOP) {
            auto prim_loop_node = cast_if<nn_ir::PrimLoopNode>(node);
            loop_nodes.push(prim_loop_node);
        } else if (node.getNodeType() == nn_compiler::nn_ir::NodeType::PRIMENDLOOP) {
            auto prim_end_loop_node = cast_if<nn_ir::PrimEndLoopNode>(node);
            auto prim_loop_node = loop_nodes.top();
            prim_end_loop_node->setGotoNode(prim_loop_node->getId());
            prim_loop_node->setGotoNode(prim_end_loop_node->getId() + 1);
            loop_nodes.pop();
        }
    }

    return RetVal::SUCCESS;
}

} // namespace nn_compiler
