/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/ir_hwnode_parser.hpp"
#include "ir/common/log.hpp"
#include "ir/ir_includes.hpp"
#include "ir/ir_tools.hpp"
#include "ir/ir_types.hpp"

namespace nn_compiler {

template <>
std::unique_ptr<nn_ir::HWNode>
IRHWNodeParser::parseHWNode<IR::HWNode::AnyType_MAAEltwiseNode>(const IR::HwNode*      ir_node,
                                                                const nn_ir::NodeInfo& node_info) {
    auto elt_node = ir_node->hw_node_as_MAAEltwiseNode();

    Log::IR::E_IF(elt_node == nullptr) << "IRHWNodeParser::parseHWNode<HW::MAAEltwiseNode>() => wrong node type!";

    nn_ir::IR_Node_Config_Type_ ir_type   = elt_node->operation();
    auto                        conf_type = nn_ir::parseConfigType(ir_type);
    nn_ir::EltwiseType          elt_type  = std::get<nn_ir::EltwiseType>(conf_type);
    auto                        act_node  = getActNode(node_info, elt_node->activation());

    // TODO(dongguen.lim): add coeff
    auto      stable_prod_grad = elt_node->stable_prod_grad();
    auto      shift_node       = getShiftNode(node_info, elt_node->shift());
    BLOB_ID_T kernel_blob_id   = elt_node->kernel_blob_id();
    BLOB_ID_T bias_blob_id     = elt_node->bias_blob_id();
    return std::make_unique<nn_ir::MAAEltwiseNode>(node_info,
                                                   elt_type,
                                                   stable_prod_grad,
                                                   std::move(shift_node),
                                                   std::move(act_node),
                                                   kernel_blob_id,
                                                   bias_blob_id);
}

std::unique_ptr<nn_ir::ActivationNode> IRHWNodeParser::getActNode(const nn_ir::NodeInfo&            node_info,
                                                                  const IR::NNNode::ActivationNode* ir_act_node) {
    if (ir_act_node == nullptr) {
        return nullptr;
    }
    nn_ir::NodeInfo node_info_(-1, node_info.name + "_activation", node_info.graph);
    auto            ir_act_type = ir_act_node->type();

    auto slope          = ir_act_node->slope();
    auto negative_slope = ir_act_node->negative_slope();
    auto min            = ir_act_node->min();
    auto max            = ir_act_node->max();
    auto shift_node     = getShiftNode(node_info, ir_act_node->shift());

    nn_ir::ActivationType act_type = parseActivation(ir_act_type);
    return std::make_unique<nn_ir::ActivationNode>(
        node_info_, std::move(shift_node), act_type, slope, negative_slope, min, max);
}

std::unique_ptr<nn_ir::ShiftNode> IRHWNodeParser::getShiftNode(const nn_ir::NodeInfo&       node_info,
                                                               const IR::OPNode::ShiftNode* ir_shift_node) {
    if (ir_shift_node == nullptr) {
        return nullptr;
    }

    nn_ir::NodeInfo shift_node_info(-1, node_info.name + "_shift", node_info.graph);
    return parseShiftNode(shift_node_info, ir_shift_node);
}

} // namespace nn_compiler
