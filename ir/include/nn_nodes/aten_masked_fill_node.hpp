#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenMaskedFillNode : public NodeMixin<AtenMaskedFillNode, NNNode>
{
   public:
    explicit AtenMaskedFillNode(const NodeInfo& node_info, bool is_inplace)
        : NodeMixin(node_info, NodeType::ATENMASKEDFILL), is_inplace_(is_inplace)
    {
    }
    std::string getNodeTypeAsString(void) const override { return "AtenMaskedFill"; }
    bool getIsInplace() const { return is_inplace_; }

   private:
    bool is_inplace_ = false;

};  // class AtenMaskedFillNode

}  // namespace nn_ir
}  // namespace nn_compiler
