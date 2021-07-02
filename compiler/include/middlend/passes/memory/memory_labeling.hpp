#pragma once

#include "compiler/include/middlend/context/compilation_context.hpp"
#include "compiler/include/middlend/context/device_label_info.hpp"
#include "compiler/include/middlend/context/memory_label_info.hpp"
#include "compiler/include/middlend/passes/pass_support.hpp"
#include "compiler/include/middlend/utils/op_basic_util.hpp"
#include "ir/include/all_nodes.hpp"
#include "ir/include/edge.hpp"
#include "ir/include/node.hpp"

namespace nn_compiler {

class MemoryLabelingPass : public PassMixin<MemoryLabelingPass> {
 public:

    RetVal initialize(const UtilManager& util_manager, const TraitManager& trait_manager);

    RetVal run(nn_ir::NNIR& graph, CompilationContext& context);

private:
    MemoryLabelInfo* memory_labeling_info_ = nullptr;
    DeviceLabelInfo* execute_labeling_info_ = nullptr;

    const OpBasicUtil* op_basic_util_      = nullptr;
}; // class MemoryLabelingPass

} // namespace nn_compiler
