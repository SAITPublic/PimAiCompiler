#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenIndexSelectNode : public NodeMixin<AtenIndexSelectNode, NNNode>
{
   public:
    explicit AtenIndexSelectNode(const NodeInfo& node_info, int64_t dim)
        : NodeMixin(node_info, NodeType::ATENINDEXSELECT), dim_(dim)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenIndexSelect"; }
    int64_t getDim() const { return dim_; }

   private:
    int64_t dim_;

};  // class AtenIndexSelectNode

}  // namespace nn_ir
}  // namespace nn_compiler
