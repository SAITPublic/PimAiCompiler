#include "common/include/common.hpp"
#include "ir/include/ir_includes.hpp"
#include "compiler/include/common/log.hpp"

#include "compiler/include/middlend/passes/graph/device_labeling.hpp"
#include "compiler/include/middlend/passes/memory/memory_labeling.hpp"

namespace nn_compiler {

RetVal MemoryLabelingPass::initialize(const UtilManager& util_manager, const TraitManager& trait_manager) {
    op_basic_util_ = util_manager.getUtil<OpBasicUtil, decltype(this)>();

    return RetVal::SUCCESS;
}

RetVal MemoryLabelingPass::run(nn_ir::NNIR& graph, CompilationContext& context) {
    execute_labeling_info_ = context.getData<DeviceLabelInfo, decltype(this)>();
    memory_labeling_info_ = context.getData<MemoryLabelInfo, decltype(this)>();

    for (auto& edge : graph.getEdges()) {
        if (edge.getOutNode() != nullptr) {
            //TODO if runtime can clssify the GPU and PIM, and PIM can read data directly from PIM memory
            memory_labeling_info_->addEdgeMemoryLabel(edge.getId(), DeviceLabelInfo::DeviceType::GPU);
        }
    }

    return RetVal::SUCCESS;
}

} // namespace nn_compiler
