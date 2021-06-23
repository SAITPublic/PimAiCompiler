#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenListNode : public NodeMixin<AtenListNode, NNNode> {
 public:
    explicit AtenListNode(const NodeInfo& node_info)
        : NodeMixin(node_info, NodeType::ATENLIST) {}

    std::string getNodeTypeAsString() const override { return "AtenList"; }
}; // class AtenListNode

} // namespace nn_ir
} // namespace nn_compiler
