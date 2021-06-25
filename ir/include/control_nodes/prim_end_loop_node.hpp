#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimEndLoopNode : public NodeMixin<PrimEndLoopNode, CONTROLNode> {
 public:
    explicit PrimEndLoopNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMENDLOOP) {}

    std::string getNodeTypeAsString(void) const override { return "PrimEndLoop"; }
}; // class PrimEndLoopNode

} // namespace nn_ir
} // namespace nn_compiler
