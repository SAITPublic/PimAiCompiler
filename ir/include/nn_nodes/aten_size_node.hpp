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
 * @file.    dummy_node.hpp
 * @brief.   This is DummyNode class
 * @details. This header defines DummyNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenSizeNode : public NodeMixin<AtenSizeNode, NNNode> {
 public:
    explicit AtenSizeNode(const NodeInfo& node_info, int64_t dim = -1)
        : NodeMixin(node_info, NodeType::ATENSIZE), dim_(dim) {}

    std::string getNodeTypeAsString() const override { return "AtenSizeNode"; }

    int64_t getDim() const { return dim_; }

 private:
    int64_t dim_;
}; // class AtenSizeNode

} // namespace nn_ir
} // namespace nn_compiler
