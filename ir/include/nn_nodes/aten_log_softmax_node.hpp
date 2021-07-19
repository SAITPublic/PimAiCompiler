#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenLogSoftmaxNode : public NodeMixin<AtenLogSoftmaxNode, NNNode>
{
   public:
    explicit AtenLogSoftmaxNode(const NodeInfo& node_info, int64_t dim, int64_t dtype)
        : NodeMixin(node_info, NodeType::ATENLOGSOFTMAX), dim_(dim), dtype_(dtype)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "AtenLogSoftmax"; }

    int64_t getDim() const { return dim_; }
    int64_t getDtype() const { return dtype_; }

   private:
    int64_t dim_;
    int64_t dtype_;
};  // class AtenLogSoftmaxNode

}  // namespace nn_ir
}  // namespace nn_compiler
