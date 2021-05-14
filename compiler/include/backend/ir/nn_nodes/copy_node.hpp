/* Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    copy_node.hpp
 * @brief.   This is CopyNode class
 * @details. This header defines CopyNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/common.hpp"
#include "common/types.hpp"

#include "ir/ir_types.hpp"
#include "ir/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class CopyNode : public NodeMixin<CopyNode, NNNode> {
 public:
    explicit CopyNode(const NodeInfo& node_info, bool is_dram_to_dram = false)
        : NodeMixin(node_info, NodeType::COPY), is_dram_to_dram_(is_dram_to_dram) {}

    std::string getNodeTypeAsString() const override { return "Copy"; }

    bool isDramToDram() const { return is_dram_to_dram_; }

 private:
    bool is_dram_to_dram_;
}; // class CopyNode

} // namespace nn_ir
} // namespace nn_compiler
