#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/control_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimDtypeNode : public NodeMixin<PrimDtypeNode, CONTROLNode> {
 public:
    explicit PrimDtypeNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMDTYPE) {}

    std::string getNodeTypeAsString(void) const override { return "PrimDtypeNode"; }

private:

}; // class PrimDtypeNode

} // namespace nn_ir
} // namespace nn_compiler
