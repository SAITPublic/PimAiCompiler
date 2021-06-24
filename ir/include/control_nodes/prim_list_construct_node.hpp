#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/control_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimListConstructNode : public NodeMixin<PrimListConstructNode, CONTROLNode> {
 public:
    explicit PrimListConstructNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMLISTCONSTRUCT) {}

    std::string getNodeTypeAsString(void) const override { return "PrimListConstruct"; }
}; // class PrimListConstructNode

} // namespace nn_ir
} // namespace nn_compiler
