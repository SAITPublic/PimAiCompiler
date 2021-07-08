#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/control_node.hpp"
#include "ir/include/ir_types.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimOutputNode : public NodeMixin<PrimOutputNode, CONTROLNode> {
 public:
    explicit PrimOutputNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMOUTPUT) {}

    std::string getNodeTypeAsString(void) const override { return "PrimOutput"; }
}; // class PrimOutputNode

} // namespace nn_ir
} // namespace nn_compiler
