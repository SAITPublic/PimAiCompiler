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
                        int64_t             dtype,
                        int                 non_blocking,
                        int                 copy,
                        int                 optional_memory_format)
        : NodeMixin(node_info, NodeType::ATENTO), dtype_(dtype), non_blocking_(non_blocking),
          copy_(copy), optional_memory_format_(optional_memory_format) {}

    std::string getNodeTypeAsString(void) const override { return "AtenTo"; }

    void setDType(int64_t dtype) { dtype_ = dtype; }

    int64_t getDType() { return dtype_; }

    void setNonBlocking(int non_blocking) { non_blocking_ = non_blocking; }

    int getNonBlocking() { return non_blocking_; }

    void setCopy(int copy) { copy_ = copy; }

    int getCopy() { return copy_; }

    void setOptionalMemoryFormat(int optional_memory_format) { optional_memory_format_ = optional_memory_format; }

    int getOptionalMemoryFormat() { return optional_memory_format_; }

 private:
    int64_t dtype_;
    int non_blocking_;
    int copy_;
    int optional_memory_format_ = -1;
}; // class AtenToNode

} // namespace nn_ir
} // namespace nn_compiler
