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

class AtenTo2Node : public NodeMixin<AtenTo2Node, NNNode> {
 public:
    explicit AtenTo2Node(const NodeInfo&     node_info,
                         int                 non_blocking,
                         int                 copy,
                         int                 optional_memory_format)
        : NodeMixin(node_info, NodeType::ATENTO2), non_blocking_(non_blocking),
          copy_(copy), optional_memory_format_(optional_memory_format) {}

    std::string getNodeTypeAsString(void) const override { return "AtenTo"; }

    void setNonBlocking(int non_blocking) { non_blocking_ = non_blocking; }

    int getNonBlocking() { return non_blocking_; }

    void setCopy(int copy) { copy_ = copy; }

    int getCopy() { return copy_; }

    void setOptionalMemoryFormat(int optional_memory_format) { optional_memory_format_ = optional_memory_format; }

    int getOptionalMemoryFormat() { return optional_memory_format_; }

 private:
    int non_blocking_;
    int copy_;
    int optional_memory_format_ = -1;
}; // class AtenTo2Node

} // namespace nn_ir
} // namespace nn_compiler
