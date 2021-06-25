#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/control_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimUninitializedNode : public NodeMixin<PrimUninitializedNode, CONTROLNode> {
 public:
    explicit PrimUninitializedNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMUNINITIALIZED) {}

    std::string getNodeTypeAsString(void) const override { return "PrimUninitialized"; }
}; // class PrimUninitializedNode

} // namespace nn_ir
} // namespace nn_compiler
