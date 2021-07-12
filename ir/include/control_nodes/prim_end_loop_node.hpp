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

    void setGotoNode(int64_t goto_node) { goto_node_ = goto_node; }

    int64_t getGotoNode() { return goto_node_; }

 private:
    int64_t goto_node_;
}; // class PrimEndLoopNode

} // namespace nn_ir
} // namespace nn_compiler
