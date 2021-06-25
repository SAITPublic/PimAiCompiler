#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenTransposeNode : public NodeMixin<AtenTransposeNode, NNNode> {
 public:
    explicit AtenTransposeNode(const NodeInfo& node_info, int64_t dim0, int64_t dim1)
        : NodeMixin(node_info, NodeType::ATENTRANSPOSE), dim0_(dim0), dim1_(dim1) {}

    std::string getNodeTypeAsString() const override { return "AtenTranspose"; }

    int64_t getDim0() const { return dim0_; }
    int64_t getDim1() const { return dim1_; }

 private:
    int64_t dim0_;
    int64_t dim1_;

}; // class AtenTransposeNode

} // namespace nn_ir
} // namespace nn_compiler
