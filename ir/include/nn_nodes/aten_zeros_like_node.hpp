
#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenZerosLikeNode : public NodeMixin<AtenZerosLikeNode, NNNode> {
 public:
    explicit AtenZerosLikeNode(const NodeInfo& node_info)
        : NodeMixin(node_info, NodeType::ATENZEROSLIKE) {}

    std::string getNodeTypeAsString() const override { return "AtenZerosLike"; }
}; // class AtenZerosLikeNode

} // namespace nn_ir
} // namespace nn_compiler
