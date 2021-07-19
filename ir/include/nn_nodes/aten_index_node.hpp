#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenIndexNode : public NodeMixin<AtenIndexNode, NNNode>
{
   public:
    explicit AtenIndexNode(const NodeInfo& node_info) : NodeMixin(node_info, NodeType::ATENINDEX) {}

    std::string getNodeTypeAsString() const override { return "AtenIndex"; }

};  // class AtenIndexNode

}  // namespace nn_ir
}  // namespace nn_compiler
