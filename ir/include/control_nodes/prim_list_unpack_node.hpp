#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/control_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimListUnpackNode : public NodeMixin<PrimListUnpackNode, CONTROLNode> {
 public:
    explicit PrimListUnpackNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMLISTUNPACK) {}

    std::string getNodeTypeAsString(void) const override { return "PrimListUnpack"; }

}; // class PrimListUnpackNode

} // namespace nn_ir
} // namespace nn_compiler
