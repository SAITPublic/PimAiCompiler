#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenNotNode : public NodeMixin<AtenNotNode, NNNode>
{
   public:
    explicit AtenNotNode(const NodeInfo& node_info) : NodeMixin(node_info, NodeType::ATENNOT) {}

    std::string getNodeTypeAsString() const override { return "AtenNot"; }
};  // class AtenNotNode

}  // namespace nn_ir
}  // namespace nn_compiler
