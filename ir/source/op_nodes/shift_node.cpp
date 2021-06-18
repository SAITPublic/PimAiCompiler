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
 * @file    shift_node.cpp
 * @brief   This is ShiftNode class
 * @details This source defines ShiftNode class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/include/op_nodes/shift_node.hpp"
#include "ir/include/common/log.hpp"

namespace nn_compiler {
namespace nn_ir {

ShiftNode::ShiftNode(const NodeInfo&       node_info,
                     std::vector<int32_t>  quantization_shift,
                     std::vector<int32_t>  multiplication_shift,
                     std::vector<int32_t>  activation_shift,
                     std::vector<uint16_t> lut_scale,
                     std::vector<uint16_t> lut_bias,
                     std::vector<uint32_t> grelu_info)
    : NodeMixin(node_info, NodeType::SHIFT), quantization_shift_(std::move(quantization_shift)),
      multiplication_shift_(std::move(multiplication_shift)), activation_shift_(std::move(activation_shift)),
      lut_scale_(std::move(lut_scale)), lut_bias_(std::move(lut_bias)), grelu_info_(std::move(grelu_info)) {}

} // namespace nn_ir
} // namespace nn_compiler
