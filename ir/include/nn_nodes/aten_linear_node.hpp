#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenLinearNode : public NodeMixin<AtenLinearNode, NNNode>
{
   public:
    explicit AtenLinearNode(const NodeInfo& node_info, std::vector<int64_t> weight_blob_ids,
                            std::vector<int64_t> bias_blob_ids)
        : NodeMixin(node_info, NodeType::ATENLINEAR), weight_blob_ids_(weight_blob_ids), bias_blob_ids_(bias_blob_ids)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "AtenLinear"; }

    std::vector<int64_t> getWeightBlobIds() const { return weight_blob_ids_; }
    std::vector<int64_t> getBiasBlobIds() const { return bias_blob_ids_; }

   private:
    std::vector<int64_t> weight_blob_ids_;
    std::vector<int64_t> bias_blob_ids_;
};  // class AtenLinearNode

}  // namespace nn_ir
}  // namespace nn_compiler
