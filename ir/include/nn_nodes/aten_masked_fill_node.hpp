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
    explicit AtenMaskedFillNode(const NodeInfo& node_info, float value)
        : NodeMixin(node_info, NodeType::ATENMASKEDFILL), value_(value)
    {
    }

    std::string getNodeTypeAsString(void) const override {
        return "AtenMaskedFill"; }

    float getValue() const {
        return value_; }

   private:
    float value_;
};  // class AtenMaskedFillNode

}  // namespace nn_ir
}  // namespace nn_compiler
