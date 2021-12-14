#include "common/include/common.hpp"
#include "ir/include/ir_includes.hpp"
#include "compiler/include/common/log.hpp"

#include "compiler/include/middlend/passes/graph/cat_labeling.hpp"

#include <set>
#include <vector>
#include <algorithm>

namespace nn_compiler {

/**
 * @brief.      initialize a CatLabelingPass
 * @param[in].  UtilManager& util_manager, TraitManager& trait_manager.
 * @param[out].
 * @returns.    return RetVal
 */
RetVal CatLabelingPass::initialize(const UtilManager& util_manager, const TraitManager& trait_manager) {
    op_basic_util_ = util_manager.getUtil<OpBasicUtil, decltype(this)>();
    return RetVal::SUCCESS;
}

/**
 * @brief.      run a CatLabelingPass
 * @param[in].  nn_ir::NNIR& graph, CompilationContext& context.
 * @param[in].
 * @param[out].
 * @returns.    return RetVal
 */
RetVal CatLabelingPass::run(nn_ir::NNIR& graph, CompilationContext& context) {
    std::vector<int64_t> target_cat_ids_bmm;
    std::vector<int64_t> target_cat_ids_lstm;
    std::vector<int64_t> target_cat_ids;
    for (auto&& node : graph.getNodes()) {
        if (node.getNodeType() == nn_ir::NodeType::ATENBMM) {
            op_basic_util_->getOffspring(target_cat_ids_bmm, graph, node, nn_ir::NodeType::ATENCAT, 3);
        }else if (node.getNodeType() == nn_ir::NodeType::ATENLSTM1 && cast<nn_ir::AtenLSTM1Node>(node).getMatchCustomOpt()) {
            op_basic_util_->getOffspring(target_cat_ids_lstm, graph, node, nn_ir::NodeType::ATENCAT, 3);
        }
    }
    for (auto cat_bmm : target_cat_ids_bmm) {
        if(std::find(target_cat_ids_lstm.begin(), target_cat_ids_lstm.end(), cat_bmm) != target_cat_ids_lstm.end()) {
            target_cat_ids.emplace_back(cat_bmm);
        }
    }
    for (auto cat_id : target_cat_ids) {
        auto cat_node = cast_if<nn_ir::AtenCatNode>(graph.getNode(cat_id));
        int mem_blob_id = cast<nn_ir::DataEdge>(cat_node->getInEdge(0)).getBlobId();
        cat_node->setMemBlobId(mem_blob_id * 10);
    }
    return RetVal::SUCCESS;
}

} // namespace nn_compiler
