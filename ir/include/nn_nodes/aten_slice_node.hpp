
#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenSliceNode : public NodeMixin<AtenSliceNode, NNNode> {
 public:
    explicit AtenSliceNode(const NodeInfo& node_info, int64_t dim, int64_t start, int64_t end, int64_t step)
        : NodeMixin(node_info, NodeType::ATENSLICE), dim_(dim), start_(start), end_(end), step_(step) {}

    std::string getNodeTypeAsString() const override { return "AtenSlice"; }

    const int64_t getDim() const {return dim_; }
    const int64_t getStart() const {return start_; }
    const int64_t getEnd() const {return end_; }
    const int64_t getStep() const {return step_; }

 private:
    int64_t dim_;
    int64_t start_;
    int64_t end_;
    int64_t step_;
}; // class AtenSliceNode

} // namespace nn_ir
} // namespace nn_compiler
