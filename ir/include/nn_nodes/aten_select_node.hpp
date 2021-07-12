#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenSelectNode : public NodeMixin<AtenSelectNode, NNNode> {
 public:
    explicit AtenSelectNode(const NodeInfo& node_info, int64_t dim, int64_t index)
            : NodeMixin(node_info, NodeType::ATENSELECT), dim_(dim), index_(index) {}

    std::string getNodeTypeAsString() const override { return "AtenSelect"; }

    void setDim(int64_t dim) { dim_ = dim; }

    int64_t getDim() { return dim_; }

    void setIndex(int64_t index) { index_ = index; }

    int64_t getIndex() { return index_; }

 private:
    int64_t dim_   = 0;
    int64_t index_ = 0;
}; // class AtenSelectNode

} // namespace nn_ir
} // namespace nn_compiler
