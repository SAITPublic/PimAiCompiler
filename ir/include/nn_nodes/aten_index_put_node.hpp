#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenIndexPutNode : public NodeMixin<AtenIndexPutNode, NNNode>
{
   public:
    explicit AtenIndexPutNode(const NodeInfo& node_info, int accumulate)
        : NodeMixin(node_info, NodeType::ATENINDEXPUT), accumulate_(accumulate)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenIndexPut"; }
    int getAccumulate() const { return accumulate_; }

   private:
    int accumulate_;

};  // class AtenIndexPutNode

}  // namespace nn_ir
}  // namespace nn_compiler
