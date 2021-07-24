/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenContiguousNode : public NodeMixin<AtenContiguousNode, NNNode> {
 public:
    explicit AtenContiguousNode(const NodeInfo& node_info, int memory_format) :
            NodeMixin(node_info, NodeType::ATENCONTIGUOUS), memory_format_(memory_format) {}

    std::string getNodeTypeAsString(void) const override { return "AtenContiguous"; }

    void setMemoryFormat(int memory_format) { memory_format_ = memory_format; }

    int getMemoryFormat() { return memory_format_; }

 private:
    int memory_format_ = INT32_MAX;

}; // class AtenContiguousNode

} // namespace nn_ir
} // namespace nn_compiler