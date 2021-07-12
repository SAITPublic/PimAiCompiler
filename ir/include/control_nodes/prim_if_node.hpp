#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimIfNode : public NodeMixin<PrimIfNode, CONTROLNode> {
 public:
    explicit PrimIfNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMIF) {}

    std::string getNodeTypeAsString(void) const override { return "PrimIf"; }

    void setElseNetStartNode(int64_t else_net_start_node) { else_net_start_node_ = else_net_start_node; }

    int64_t getElseNetStartNode() { return else_net_start_node_; }

 private:
    int64_t else_net_start_node_;
}; // class PrimIfNode

} // namespace nn_ir
} // namespace nn_compiler
