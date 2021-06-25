#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenSelectNode : public NodeMixin<AtenSelectNode, NNNode> {
 public:
    explicit AtenSelectNode(const NodeInfo& node_info)
        : NodeMixin(node_info, NodeType::ATENSELECT) {}

    std::string getNodeTypeAsString() const override { return "AtenSelect"; }

}; // class AtenSelectNode

} // namespace nn_ir
} // namespace nn_compiler
