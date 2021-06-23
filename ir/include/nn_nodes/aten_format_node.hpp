/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file.    aten_format_node.hpp
 * @brief.   This is AtenFormatNode class
 * @details. This header defines AtenFormatNode class.
 * @version. 0.1.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenFormatNode : public NodeMixin<AtenFormatNode, NNNode> {
 public:
    explicit AtenFormatNode(const NodeInfo& node_info,  std::string assembly_format)
        : NodeMixin(node_info, NodeType::ATENFORMAT), assembly_format_(assembly_format) {}

    std::string getNodeTypeAsString() const override { return "AtenFormat"; }
    std::string getAssemblyFormat() const { return assembly_format_; }

 private:
    std::string assembly_format_;
}; // class AtenFormatNode

} // namespace nn_ir
} // namespace nn_compiler
