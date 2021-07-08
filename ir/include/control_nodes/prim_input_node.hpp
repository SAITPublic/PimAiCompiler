#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimInputNode : public NodeMixin<PrimInputNode, CONTROLNode> {
 public:
    explicit PrimInputNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMINPUT) {}

    std::string getNodeTypeAsString(void) const override { return "PrimInput"; }
}; // class PrimInputNode

} // namespace nn_ir
} // namespace nn_compiler
