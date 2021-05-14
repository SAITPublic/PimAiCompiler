/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/nn_nodes/convolution_node.hpp"
#include "ir/common/log.hpp"

#include "common/arithmetics.hpp"

namespace nn_compiler {
namespace nn_ir {

ConvolutionNode::ConvolutionNode(const NodeInfo&                 node_info,
                                 std::unique_ptr<ActivationNode> activation_node,
                                 std::unique_ptr<ShiftNode>      shift_node,
                                 Shape2D                         kernel_size,
                                 Shape2D                         stride_size,
                                 Shape2D                         dilation_size,
                                 Pad4                            padding_size,
                                 BLOB_ID_T                       kernel_blob_id,
                                 BLOB_ID_T                       bias_blob_id)
    : NodeMixin(node_info, NodeType::CONVOLUTION), activation_node_(std::move(activation_node)),
      shift_node_(std::move(shift_node)),
      kernel_node_parameters_(padding_size, kernel_size, stride_size, dilation_size), kernel_blob_id_(kernel_blob_id),
      bias_blob_id_(bias_blob_id) {}

ConvolutionNode::ConvolutionNode(const ConvolutionNode& copied_conv)
    : NodeMixin(copied_conv),
      activation_node_(copied_conv.activation_node_ ? copied_conv.activation_node_->clone() : nullptr),
      shift_node_(copied_conv.shift_node_ ? copied_conv.shift_node_->clone() : nullptr),
      kernel_node_parameters_(copied_conv.kernel_node_parameters_), kernel_blob_id_(copied_conv.kernel_blob_id_),
      bias_blob_id_(copied_conv.bias_blob_id_) {}

} // namespace nn_ir
} // namespace nn_compiler
