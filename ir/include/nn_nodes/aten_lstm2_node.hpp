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

#include "ir/include/ir_includes.hpp"

namespace nn_compiler {
namespace nn_ir {

class AtenLSTM2Node : public NodeMixin<AtenLSTM2Node, NNNode> {
 public:
    explicit AtenLSTM2Node(const NodeInfo&                 node_info,
                             int                         has_biases,
                             int64_t                      num_layer,
                             double                         dropout,
                             int                              train,
                             int                      bidirectional,
                             std::vector<int64_t>              weight_blob_ids,
                             std::vector<int64_t>                bias_blob_ids)
     : NodeMixin(node_info, NodeType::ATENLSTM2), has_biases_(has_biases), num_layers_(num_layer),
                 dropout_(dropout), train_(train), bidirectional_(bidirectional),
                 weight_blob_ids_(weight_blob_ids), bias_blob_ids_(bias_blob_ids) {}


    std::string getNodeTypeAsString() const override { return "AtenLSTM2"; }

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

    void setHasBiases(int has_biases) { has_biases_ = has_biases; }
    void setNumLayers(int64_t num_layers) { num_layers_ = num_layers;}
    void setDropout(double dropout) { dropout_ = dropout; }
    void setTrain(int train) { train_ = train; }
    void setBidirectional(int bidirectional) { bidirectional_ = bidirectional; }

    int     getHasBiases() { return has_biases_; }
    int64_t  getNumLayers() { return num_layers_; }
    double   getDropout() { return dropout_; }
    int     getTrain() { return train_; }
    int     getBidirectional() { return bidirectional_; }

    std::vector<Shape4D> getPreprocessedWeightBlobDim() const override {
       std::vector<Shape4D> weight_blob_shapes;
       for (auto weight_blob_id : weight_blob_ids_) {
          auto weight_blob_shape = getGraph().getBlob(weight_blob_id)->getShape();
          weight_blob_shapes.push_back(weight_blob_shape);
       }
       return weight_blob_shapes;
    }

 private:
    int             has_biases_;
    int64_t         num_layers_;
    double             dropout_;
    int                  train_;
    int          bidirectional_;

    std::vector<int64_t>       weight_blob_ids_;
    std::vector<int64_t>         bias_blob_ids_;
}; // class AtenLSTM2Node

} // namespace nn_ir
} // namespace nn_compiler
