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
 * @file    data_format_node.cpp
 * @brief   This is DataFormatNode class
 * @details This source defines DataFormatNode class.
 * @version 0.1 ... (version comment)
 *              Supported functions are as follow:
 */

#include "ir/include/nn_nodes/data_format_node.hpp"
#include "ir/include/common/log.hpp"

namespace nn_compiler {
namespace nn_ir {

DataFormatNode::DataFormatNode(const NodeInfo& node_info, DataFormatConversion format_direction, Shape4D shape)
    : NodeMixin(node_info, NodeType::DATAFORMAT), format_direction_(format_direction), shape_(shape) {}

} // namespace nn_ir
} // namespace nn_compiler
