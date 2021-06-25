#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenCatNode : public NodeMixin<AtenCatNode, NNNode> {
 public:
    explicit AtenCatNode(const NodeInfo& node_info, int64_t dim)
        : NodeMixin(node_info, NodeType::ATENCAT), dim_(dim) {}

    std::string getNodeTypeAsString() const override { return "AtenCat"; }

    int64_t getDim() const { return dim_; }

 private:
    int64_t dim_;

}; // class AtenCatNode

} // namespace nn_ir
} // namespace nn_compiler
