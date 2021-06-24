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

class AtenToNode : public NodeMixin<AtenToNode, NNNode> {
 public:
    explicit AtenToNode(const NodeInfo&     node_info,
                        DataType            dtype,
                        bool                non_blocking,
                        bool                copy,
                        bool                optional_memory_format)
        : NodeMixin(node_info, NodeType::ATENTO), dtype_(dtype), non_blocking_(non_blocking),
          copy_(copy), optional_memory_format_(optional_memory_format) {}

    std::string getNodeTypeAsString(void) const override { return "AtenTo"; }

 private:
    DataType dtype_;
    bool non_blocking_ = false;
    bool copy_ = false;
    int64_t optional_memory_format_ = 0;
}; // class AtenToNode

} // namespace nn_ir
} // namespace nn_compiler
