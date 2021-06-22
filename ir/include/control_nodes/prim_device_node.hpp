#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/control_node.hpp"

namespace nn_compiler {
namespace nn_ir {

class PrimDeviceNode : public NodeMixin<PrimDeviceNode, CONTROLNode> {
 public:
    explicit PrimDeviceNode(const NodeInfo &node_info)
            : NodeMixin(node_info, NodeType::PRIMDEVICE) {}

    std::string getNodeTypeAsString(void) const override { return "PrimDeviceNode"; }

private:

}; // class PrimDeviceNode

} // namespace nn_ir
} // namespace nn_compiler
