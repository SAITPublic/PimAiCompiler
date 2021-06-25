#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/control_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimEndIfNode : public NodeMixin<PrimEndIfNode, CONTROLNode> {
 public:
    explicit PrimEndIfNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMENDIF) {}

    std::string getNodeTypeAsString(void) const override { return "PrimEndIf"; }

}; // class PrimEndIfNode

} // namespace nn_ir
} // namespace nn_compiler
