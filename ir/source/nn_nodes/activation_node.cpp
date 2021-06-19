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
 * @file    activation_node.cpp
 * @brief   This is ActivationNode class
 * @details This source defines ActivationNode class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/include/nn_nodes/activation_node.hpp"
#include "ir/include/common/log.hpp"

namespace nn_compiler {
namespace nn_ir {
ActivationNode::ActivationNode(const ActivationNode& other)
    : NodeMixin(other), shift_node_(other.shift_node_ ? other.shift_node_->clone() : nullptr),
      activation_type_(other.activation_type_), slope_(other.slope_), negative_slope_(other.negative_slope_),
      min_(other.min_), max_(other.max_) {}
} // namespace nn_ir
} // namespace nn_compiler