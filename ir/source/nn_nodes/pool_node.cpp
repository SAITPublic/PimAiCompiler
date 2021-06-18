/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "ir/nn_nodes/pool_node.hpp"
#include "ir/common/log.hpp"

namespace nn_compiler {
namespace nn_ir {

PoolNode::PoolNode(const NodeInfo& node_info,
                   PoolType        pool_type,
                   Shape2D         kernel_size,
                   Shape2D         stride_size,
                   Shape2D         dilation_size,
                   Pad4            padding_size,
                   PadCalcType     pad_calc_type)
    : NodeMixin(node_info, NodeType::POOL), pool_type_(pool_type),
      kernel_node_parameters_(padding_size, kernel_size, stride_size, dilation_size), pad_calc_type_(pad_calc_type) {}

} // namespace nn_ir
} // namespace nn_compiler
