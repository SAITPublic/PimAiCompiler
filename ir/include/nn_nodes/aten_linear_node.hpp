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
    std::vector<Blob*> getWeightBlob() const
    {
        std::vector<Blob*> weight_blobs;
        for (auto weight_blob_id : weight_blob_ids_) {
            auto weight_blob = getGraph().getBlob(weight_blob_id);
            weight_blobs.push_back(weight_blob);
        }
        return weight_blobs;
    }

    std::vector<Blob*> getBiasBlob() const
    {
        std::vector<Blob*> bias_blobs;
        for (auto bias_blob_id : bias_blob_ids_) {
            auto bias_blob = (bias_blob_id == INVALID_ID) ? nullptr : getGraph().getBlob(bias_blob_id);
            bias_blobs.push_back(bias_blob);
        }
        return bias_blobs;
    }

   private:
    std::vector<int64_t> weight_blob_ids_;
    std::vector<int64_t> bias_blob_ids_;
};  // class AtenLinearNode

}  // namespace nn_ir
}  // namespace nn_compiler
