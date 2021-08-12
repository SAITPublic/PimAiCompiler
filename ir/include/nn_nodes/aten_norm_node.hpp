#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenNormNode : public NodeMixin<AtenNormNode, NNNode>
{
   public:
    explicit AtenNormNode(const NodeInfo& node_info, int64_t p) : NodeMixin(node_info, NodeType::ATENNORM), p_(p) {}

    std::string getNodeTypeAsString() const override { return "AtenNorm"; }

    void setP(int64_t p) { p_ = p; }

    int64_t getP() { return p_; }

private:
    int64_t p_;
};  // class AtenNormNode

}  // namespace nn_ir
}  // namespace nn_compiler
