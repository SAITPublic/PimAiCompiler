/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/hw_nodes/maa_eltwise_node.hpp"
#include "ir/common/log.hpp"

namespace nn_compiler::nn_ir {

MAAEltwiseNode::MAAEltwiseNode(const MAAEltwiseNode& copied_eltwise)
    : NodeMixin(copied_eltwise), elt_type_(copied_eltwise.elt_type_),
      stable_prod_grad_(copied_eltwise.stable_prod_grad_),
      shift_node_(copied_eltwise.shift_node_ ? copied_eltwise.shift_node_->clone() : nullptr),
      activation_node_(copied_eltwise.activation_node_ ? copied_eltwise.activation_node_->clone() : nullptr),
      kernel_blob_id_(copied_eltwise.getKernelBlobId()), bias_blob_id_(copied_eltwise.getBiasBlobId()) {}

MAAEltwiseNode::MAAEltwiseNode(EltwiseNode& copied_eltwise)
    : NodeMixin(copied_eltwise), elt_type_(copied_eltwise.getEltType()),
      stable_prod_grad_(copied_eltwise.getStableProdGrad()),
      kernel_node_parameters_(copied_eltwise.getKernelNodeParameters()),
      shift_node_(copied_eltwise.getShiftNode() ? copied_eltwise.getShiftNode()->clone() : nullptr),
      activation_node_(nullptr), kernel_blob_id_(INVALID_ID), bias_blob_id_(INVALID_ID) {
    setType(NodeType::MAAELTWISE);
}
} // namespace nn_compiler::nn_ir
