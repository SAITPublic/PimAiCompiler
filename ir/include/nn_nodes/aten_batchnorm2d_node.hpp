/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include "common/include/common.hpp"
#include "common/include/types.hpp"

#include "ir/include/ir_types.hpp"
#include "ir/include/nn_node.hpp"

namespace nn_compiler
{
namespace nn_ir
{
class AtenBatchNorm2dNode : public NodeMixin<AtenBatchNorm2dNode, NNNode>
{
   public:
    explicit AtenBatchNorm2dNode(const NodeInfo& node_info, std::vector<int64_t> weight_blob_ids,
                                 std::vector<int64_t> bias_blob_ids, int training, double momentum, double eps,
                                 int cudnn_enable)
        :

          NodeMixin(node_info, NodeType::ATENBATCHNORM2D),
          weight_blob_ids_(weight_blob_ids),
          bias_blob_ids_(bias_blob_ids),
          training_(training),
          momentum_(momentum),
          eps_(eps),
          cudnn_enable_(cudnn_enable)
    {
    }

    std::string getNodeTypeAsString(void) const override { return "AtenBatchNorm2d"; }

    void setWeightBlobId(std::vector<int64_t> id) { weight_blob_ids_ = id; }
    void setBiasBlobId(std::vector<int64_t> id) { bias_blob_ids_ = id; }

    std::vector<int64_t> getWeightBlobId() const { return weight_blob_ids_; }
    std::vector<int64_t> getBiasBlobId() const { return bias_blob_ids_; }

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

    void setTraining(int training) { training_ = training; }

    int getTraining() const { return training_; }

    void setMomentum(double momentum) { momentum_ = momentum; }

    double getMomentum() const { return momentum_; }

    void setEps(double eps) { eps_ = eps; }

    double getEps() const { return eps_; }

    void setCudnnEnable(int cudnn_enable) { cudnn_enable_ = cudnn_enable; }

    int getCudnnEnable() { return cudnn_enable_; }

    std::vector<Shape4D> getPreprocessedWeightBlobDim() const override
    {
        std::vector<Shape4D> weight_blob_shapes;
        for (auto weight_blob_id : weight_blob_ids_) {
            auto weight_blob_shape = getGraph().getBlob(weight_blob_id)->getShape();
            weight_blob_shapes.push_back(weight_blob_shape);
        }
        return weight_blob_shapes;
    }

   private:
    std::vector<int64_t> weight_blob_ids_;
    std::vector<int64_t> bias_blob_ids_;
    int32_t training_;
    double momentum_;
    double eps_;
    int32_t cudnn_enable_;

};  // class AtenBatchNorm2dNode

}  // namespace nn_ir
}  // namespace nn_compiler
