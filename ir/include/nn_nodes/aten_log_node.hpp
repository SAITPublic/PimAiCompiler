#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenLogNode : public NodeMixin<AtenLogNode, NNNode>
{
   public:
    explicit AtenLogNode(const NodeInfo& node_info) : NodeMixin(node_info, NodeType::ATENLOG) {}

    std::string getNodeTypeAsString(void) const override { return "AtenLog"; }
};  // class AtenLogNode

}  // namespace nn_ir
}  // namespace nn_compiler
