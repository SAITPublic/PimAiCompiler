#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenGatherNode : public NodeMixin<AtenGatherNode, NNNode>
{
   public:
    explicit AtenGatherNode(const NodeInfo& node_info, int64_t dim, int sparse_grad)
        : NodeMixin(node_info, NodeType::ATENGATHER), dim_(dim), sparse_grad_(sparse_grad)
    {
    }

    std::string getNodeTypeAsString() const override { return "AtenGather"; }
    int64_t getDim() const { return dim_; }
    int getSparseGrad() const { return sparse_grad_; }

   private:
    int64_t dim_;
    int sparse_grad_;
};  // class AtenGatherNode

}  // namespace nn_ir
}  // namespace nn_compiler
