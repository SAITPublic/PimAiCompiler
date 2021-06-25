#pragma once

#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/context/device_label_info.hpp"
#include "compiler/include/middlend/passes/pass_support.hpp"
#include "compiler/include/middlend/utils/op_basic_util.hpp"
#include "ir/include/all_nodes.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/node.hpp"

namespace nn_compiler {

class DeviceLabelingPass : public PassMixin<DeviceLabelingPass> {
 public:
    /**
     * @brief.      initialize a DeviceLabelingPass
     * @param[in].  UtilManager& util_manager, TraitManager& trait_manager.
     * @param[out].
     * @returns.    return RetVal
     */
    RetVal initialize(const UtilManager& util_manager, const TraitManager& trait_manager);

    /**
     * @brief.      run a DeviceLabelingPass
     * @param[in].  std::vector<const nn_ir::NNIR&> graph, CompilationContext context.
     * @param[out].
     * @returns.    return RetVal
     */
    RetVal run(nn_ir::NNIR& graph, CompilationContext& context);

private:
    DeviceLabelInfo* device_labeling_info_ = nullptr;

    const OpBasicUtil* op_basic_util_      = nullptr;

}; // class DeviceLabelingPass

} // namespace nn_compiler
