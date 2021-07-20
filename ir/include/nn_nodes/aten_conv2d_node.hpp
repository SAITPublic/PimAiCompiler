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

namespace nn_compiler {
namespace nn_ir {

class AtenConv2dNode : public NodeMixin<AtenConv2dNode, NNNode> {
 public:
    explicit AtenConv2dNode(const NodeInfo& node_info,
                            std::vector<int64_t> weight_blob_ids,
                            std::vector<int64_t> bias_blob_ids,
                            Shape2D stride, Pad4 padding, Shape2D dilation, int64_t groups) :
            NodeMixin(node_info, NodeType::ATENCONV2D), weight_blob_ids_(weight_blob_ids),
            bias_blob_ids_(bias_blob_ids), stride_(stride), padding_(padding), dilation_(dilation), groups_(groups) {}

    std::string getNodeTypeAsString(void) const override { return "AtenConv2d"; }

    void setWeightBlobId(std::vector<int64_t > id) { weight_blob_ids_ = id; }
    void setBiasBlobId(std::vector<int64_t > id) { bias_blob_ids_ = id; }

    std::vector<int64_t> getWeightBlobId() const { return weight_blob_ids_; }
    std::vector<int64_t> getBiasBlobId() const { return bias_blob_ids_; }

    std::vector<Blob*> getWeightBlob() const {
        std::vector<Blob*> weight_blobs;
        for (auto weight_blob_id : weight_blob_ids_) {
            auto weight_blob = getGraph().getBlob(weight_blob_id);
            weight_blobs.push_back(weight_blob);
        }
        return weight_blobs;
    }

    std::vector<Blob*> getBiasBlob() const {
        std::vector<Blob*> bias_blobs;
        for (auto bias_blob_id : bias_blob_ids_) {
            auto bias_blob = (bias_blob_id == INVALID_ID) ? nullptr : getGraph().getBlob(bias_blob_id);
            bias_blobs.push_back(bias_blob);
        }
        return bias_blobs;
    }

    void setPadding(Pad4 padding) { padding_ = padding; }

    Pad4 getPadding() const { return padding_; }

    void setStride(Shape2D stride) { stride_ = stride; }

    Shape2D getStride() const { return stride_; }

    void setDilation(Shape2D dilation) { dilation_ = dilation; }

    Shape2D getDilation() const { return dilation_; }

    void setGroups(int64_t groups) { groups_ = groups; }

    int64_t getGroups() const { return groups_; }

    std::vector<Shape4D> getPreprocessedWeightBlobDim() const override {
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
    Shape2D stride_ = {{.h = 1, .w = 1}};
    Pad4 padding_ = {0, 0, 0, 0};
    Shape2D dilation_ = {{.h = 1, .w = 1}};
    int64_t groups_ = INT64_MAX;

}; // class AtenConv2dNode

} // namespace nn_ir
} // namespace nn_compiler
