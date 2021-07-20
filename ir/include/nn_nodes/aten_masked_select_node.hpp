#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenMaskedSelectNode : public NodeMixin<AtenMaskedSelectNode, NNNode>
{
   public:
    explicit AtenMaskedSelectNode(const NodeInfo& node_info) : NodeMixin(node_info, NodeType::ATENMASKEDSELECT) {}

    std::string getNodeTypeAsString(void) const override { return "AtenMaskedSelect"; }
};  // class AtenMaskedSelectNode

}  // namespace nn_ir
}  // namespace nn_compiler
