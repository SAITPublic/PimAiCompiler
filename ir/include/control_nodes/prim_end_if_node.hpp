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

    void setGotoNode(int64_t goto_node) { goto_node_ = goto_node; }
    void setIsElseNet(bool is_else_net) { is_else_net_ = is_else_net; }

    int64_t getGotoNode() { return goto_node_; }
    bool getIsElseNet() { return is_else_net_; }
 private:
    int64_t goto_node_;
    bool is_else_net_ = false;
}; // class PrimEndIfNode

} // namespace nn_ir
} // namespace nn_compiler
