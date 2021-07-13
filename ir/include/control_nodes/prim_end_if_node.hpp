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
    void setIfNodeId(int64_t if_node_id) { if_node_id_ = if_node_id; }

    int64_t getGotoNode() const { return goto_node_; }
    bool getIsElseNet() const { return is_else_net_; }
    int64_t getIfNodeId() const { return if_node_id_; }

 private:
    int64_t goto_node_;
    bool is_else_net_ = false;
    int64_t if_node_id_;
}; // class PrimEndIfNode

} // namespace nn_ir
} // namespace nn_compiler
