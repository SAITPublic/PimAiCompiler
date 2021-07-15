#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimLoopIndexNode : public NodeMixin<PrimLoopIndexNode, CONTROLNode> {
 public:
    explicit PrimLoopIndexNode(const NodeInfo &node_info, int64_t index)
            : NodeMixin(node_info, NodeType::PRIMLOOPINDEX), index_(index) {}

    std::string getNodeTypeAsString(void) const override { return "PrimLoopIndex"; }

    void setIndex(int64_t index) { index_ = index; }

    int64_t getIndex() const { return index_; }

 private:
    int64_t index_ = 0;
}; // class PrimLoopIndexNode

} // namespace nn_ir
} // namespace nn_compiler
