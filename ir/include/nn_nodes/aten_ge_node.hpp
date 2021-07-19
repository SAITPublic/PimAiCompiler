#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenGeNode : public NodeMixin<AtenGeNode, NNNode>
{
   public:
    explicit AtenGeNode(const NodeInfo& node_info) : NodeMixin(node_info, NodeType::ATENGE) {}

    std::string getNodeTypeAsString() const override { return "AtenGe"; }

};  // class AtenGeNode

}  // namespace nn_ir
}  // namespace nn_compiler
