#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenMulNode : public NodeMixin<AtenMulNode, NNNode>
{
   public:
    explicit AtenMulNode(const NodeInfo& node_info, float other)
        : NodeMixin(node_info, NodeType::ATENMUL), other_(other)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenMul"; }

    float getOther() const { return other_; }

   private:
    float other_;
};  // class AtenMulNode

}  // namespace nn_ir
}  // namespace nn_compiler
