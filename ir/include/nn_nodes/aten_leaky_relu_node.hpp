#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenLeakyReluNode : public NodeMixin<AtenLeakyReluNode, NNNode>
{
   public:
    explicit AtenLeakyReluNode(const NodeInfo& node_info, double scalar)
        : NodeMixin(node_info, NodeType::ATENLEAKYRELU), scalar_(scalar)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "AtenLeakyRelu"; }

    double getScalar() const { return scalar_; }

   private:
    double scalar_;
};  // class AtenLeakyReluNode

}  // namespace nn_ir
}  // namespace nn_compiler
