#include "common/include/common.hpp"
#include "ir/include/ir_includes.hpp"
#include "compiler/include/common/log.hpp"

#include "compiler/include/middlend/passes/graph/device_labeling.hpp"

namespace nn_compiler {

/**
 * @brief.      initialize a DeviceLabelingPass
 * @param[in].  UtilManager& util_manager, TraitManager& trait_manager.
 * @param[out].
 * @returns.    return RetVal
 */
RetVal DeviceLabelingPass::initialize(const UtilManager& util_manager, const TraitManager& trait_manager) {
    op_basic_util_ = util_manager.getUtil<OpBasicUtil, decltype(this)>();

    return RetVal::SUCCESS;
}

/**
 * @brief.      run a DeviceLabelingPass
 * @param[in].  nn_ir::NNIR& graph, CompilationContext& context.
 * @param[in].
 * @param[out].
 * @returns.    return RetVal
 */
RetVal DeviceLabelingPass::run(nn_ir::NNIR& graph, CompilationContext& context) {
    device_labeling_info_ = context.getData<DeviceLabelInfo, decltype(this)>();
    for (auto&& node : graph.getNodes()) {
        if (op_basic_util_->isAtenOp(node)) {
            if (auto aten_lstm1_node = cast_if<nn_ir::AtenLSTM1Node>(node)) {
                device_labeling_info_->addOpDeviceLabel(op_basic_util_->getAtenOpName(node),
                                                        DeviceLabelInfo::DeviceType::PIM);
                aten_lstm1_node->setTargetDevice("PIM");
            } else if(auto aten_lstm2_node = cast_if<nn_ir::AtenLSTM2Node>(node)) {
                device_labeling_info_->addOpDeviceLabel(op_basic_util_->getAtenOpName(node),
                                                        DeviceLabelInfo::DeviceType::PIM);
                aten_lstm2_node->setTargetDevice("PIM");
            } else if (auto aten_matmul_node = cast_if<nn_ir::AtenMatmulNode>(node)) {
                // TODO(SRCX) : for aten::matmul, only GEMV runs on PIM. Need to fix when Op shapes are fixed.
                device_labeling_info_->addOpDeviceLabel(op_basic_util_->getAtenOpName(node),
                                                        DeviceLabelInfo::DeviceType::PIM);
                aten_matmul_node->setTargetDevice("PIM");
            } else {
                device_labeling_info_->addOpDeviceLabel(op_basic_util_->getAtenOpName(node),
                                                        DeviceLabelInfo::DeviceType::GPU);
                auto aten_node = cast_if<nn_ir::NNNode>(node);
                aten_node->setTargetDevice("GPU");
            }
        } else if (op_basic_util_->isPrimOp(node)) {
            device_labeling_info_->addOpDeviceLabel(op_basic_util_->getPrimOpName(node),
                                                    DeviceLabelInfo::DeviceType::CPU);
            auto prim_node = cast_if<nn_ir::CONTROLNode>(node);
            prim_node->setTargetDevice("CPU");
        } else {
            Log::ME::D() << "Unsupported Op occurs in DeviceLabelingPass.";
            return RetVal::FAILURE;
        }
    }
    return RetVal::SUCCESS;
}

} // namespace nn_compiler
