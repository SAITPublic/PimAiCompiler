#include "common/include/common.hpp"
#include "ir/include/ir_includes.hpp"
#include "compiler/include/common/log.hpp"

#include "compiler/include/middlend/passes/graph/lstm_labeling.hpp"

#include <set>
#include <vector>

namespace nn_compiler {

/**
 * @brief.      initialize a LstmLabelingPass
 * @param[in].  UtilManager& util_manager, TraitManager& trait_manager.
 * @param[out].
 * @returns.    return RetVal
 */
RetVal LstmLabelingPass::initialize(const UtilManager& util_manager, const TraitManager& trait_manager) {
    return RetVal::SUCCESS;
}

/**
 * @brief.      run a LstmLabelingPass
 * @param[in].  nn_ir::NNIR& graph, CompilationContext& context.
 * @param[in].
 * @param[out].
 * @returns.    return RetVal
 */
RetVal LstmLabelingPass::run(nn_ir::NNIR& graph, CompilationContext& context) {
    int custom_opt_number = 0;

    for (auto&& node : graph.getNodes()) {
        if (node.getNodeType() == nn_ir::NodeType::ATENLSTM1) {
            std::set<nn_ir::Node*> tuple_construct_nodes;
            int to_tuple_num = 0;
            auto lstm_out_edges_id = node.getOutEdgeIds();
            for (auto out_id : lstm_out_edges_id) {
                auto out_node = graph.getEdge(out_id)->getOutNode();
                if (out_node != nullptr && out_node->getNodeType() == nn_ir::NodeType::PRIMTUPLECONSTRUCT) {
                    tuple_construct_nodes.insert(out_node);
                    to_tuple_num++;
                }
            }
            // two outputs from lstm1 node come to a single tuple_construct node.
            if (to_tuple_num == 2 && tuple_construct_nodes.size() == 1) {
                auto tuple_construct_node = *tuple_construct_nodes.begin();
                auto tuple_construct_out_edges_id = tuple_construct_node->getOutEdgeIds();
                // this tuple will be sent to aten::append only.
                if (tuple_construct_out_edges_id.size() != 1) {
                    continue;
                }
                auto out_node = graph.getEdge(tuple_construct_out_edges_id[0])->getOutNode();
                if (out_node != nullptr && out_node->getNodeType() == nn_ir::NodeType::ATENAPPEND) {
                    auto lstm_node = cast_if<nn_ir::AtenLSTM1Node>(node);
                    lstm_node->setMatchCustomOpt(true);
                    lstm_node->setCustomOptNumber(custom_opt_number++);
                }
            }
        }
    }
    return RetVal::SUCCESS;
}

} // namespace nn_compiler
