/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    eltwise_node.cpp
 * @brief   This is EltwiseNode class
 * @details This source defines EltwiseNode class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/nn_nodes/eltwise_node.hpp"
#include "ir/common/log.hpp"

namespace nn_compiler::nn_ir {

EltwiseNode::EltwiseNode(const EltwiseNode& copied_eltwise)
    : NodeMixin(copied_eltwise), elt_type_(copied_eltwise.elt_type_),
      stable_prod_grad_(copied_eltwise.stable_prod_grad_),
      shift_node_(copied_eltwise.shift_node_ ? copied_eltwise.shift_node_->clone() : nullptr),
      shift_in1_node_(copied_eltwise.shift_in1_node_ ? copied_eltwise.shift_in2_node_->clone() : nullptr),
      shift_in2_node_(copied_eltwise.shift_in2_node_ ? copied_eltwise.shift_in2_node_->clone() : nullptr),
      multi_scale_(copied_eltwise.multi_scale_) {}

} // namespace nn_compiler::nn_ir
