#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimBlockNode : public NodeMixin<PrimBlockNode, CONTROLNode> {
 public:
    explicit PrimBlockNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMBLOCK) {}

    std::string getNodeTypeAsString(void) const override { return "PrimBlock"; }
}; // class PrimBlockNode

} // namespace nn_ir
} // namespace nn_compiler
