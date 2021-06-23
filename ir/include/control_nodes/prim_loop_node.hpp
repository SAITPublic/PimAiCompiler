#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimLoopNode : public NodeMixin<PrimLoopNode, CONTROLNode> {
 public:
    explicit PrimLoopNode(const NodeInfo &node_info, int64_t trip_count, bool cond)
            : NodeMixin(node_info, NodeType::PRIMLOOP), trip_count_(trip_count), cond_(cond) {}

    std::string getNodeTypeAsString(void) const override { return "PrimLoopNode"; }

    void setTripCount(const int64_t trip_count) { trip_count_ = trip_count; }
    void setCond(bool cond) { cond_ = cond; }

    const int64_t getTripCount() { return trip_count_; }
    bool getCond() { return cond_; }

 private:
    int64_t trip_count_ = 0x7FFFFFFFFFFFFFFF;
    bool cond_ = true;
}; // class PrimLoopNode

} // namespace nn_ir
} // namespace nn_compiler
