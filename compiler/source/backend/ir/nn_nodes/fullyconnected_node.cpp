/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/nn_nodes/fullyconnected_node.hpp"
#include "ir/common/log.hpp"

#include "common/arithmetics.hpp"

namespace nn_compiler {
namespace nn_ir {

FullyConnectedNode::FullyConnectedNode(const NodeInfo&                 node_info,
                                       std::unique_ptr<ActivationNode> activation_node,
                                       std::unique_ptr<ShiftNode>      shift_node,
                                       nn_ir::Axis                     axis,
                                       bool                            transpose,
                                       BLOB_ID_T                       weight_blob_id,
                                       BLOB_ID_T                       bias_blob_id)
    : NodeMixin(node_info, NodeType::FULLYCONNECTED), activation_node_(std::move(activation_node)),
      shift_node_(std::move(shift_node)), axis_(axis), transpose_(transpose), kernel_blob_id_(weight_blob_id),
      bias_blob_id_(bias_blob_id) {}

FullyConnectedNode::FullyConnectedNode(const FullyConnectedNode& copied_fc)
    : NodeMixin(copied_fc),
      activation_node_(copied_fc.activation_node_ ? copied_fc.activation_node_->clone() : nullptr),
      shift_node_(copied_fc.shift_node_ ? copied_fc.shift_node_->clone() : nullptr), axis_(copied_fc.axis_),
      transpose_(copied_fc.transpose_), kernel_blob_id_(copied_fc.kernel_blob_id_),
      bias_blob_id_(copied_fc.bias_blob_id_) {}

} // namespace nn_ir
} // namespace nn_compiler
