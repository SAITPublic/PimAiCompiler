#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenMinNode : public NodeMixin<AtenMinNode, NNNode>
{
   public:
    explicit AtenMinNode(const NodeInfo& node_info, int64_t dim_or_y, int keep_dim)
        : NodeMixin(node_info, NodeType::ATENMIN), dim_or_y_(dim_or_y), keep_dim_(keep_dim)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "AtenMin"; }
    int64_t getDimOrY() const { return dim_or_y_; }
    int getKeepDim() const { return keep_dim_; }

   private:
    int64_t dim_or_y_;
    int keep_dim_;
};  // class AtenMinNode

}  // namespace nn_ir
}  // namespace nn_compiler
