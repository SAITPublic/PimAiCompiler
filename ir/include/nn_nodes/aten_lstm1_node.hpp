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

class AtenLSTM1Node : public NodeMixin<AtenLSTM1Node, NNNode> {
 public:
    explicit AtenLSTM1Node(const NodeInfo&        node_info,
                             int                  has_biases,
                             int64_t              num_layer,
                             double               dropout,
                             int                  train,
                             int                  bidirectional,
                             int                  batch_first,
                             std::vector<int64_t> weight_blob_ids,
                             std::vector<int64_t> bias_blob_ids,
                             bool                 match_custom_opt = false,
                             int                  custom_opt_number = 0)
     : NodeMixin(node_info, NodeType::ATENLSTM1), has_biases_(has_biases), num_layers_(num_layer),
                 dropout_(dropout), train_(train), bidirectional_(bidirectional), batch_first_(batch_first),
                 weight_blob_ids_(weight_blob_ids), bias_blob_ids_(bias_blob_ids),
                 match_custom_opt_(match_custom_opt),  custom_opt_number_(custom_opt_number) {}


    std::string getNodeTypeAsString() const override { return "AtenLSTM1"; }

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
    void setBatchFirst(int batch_first) { batch_first_ = batch_first; }
    void setMatchCustomOpt(bool match_custom_opt) { match_custom_opt_ = match_custom_opt; }
    void setCustomOptNumber(int custom_opt_number) { custom_opt_number_ = custom_opt_number; }

    int     getHasBiases() { return has_biases_; }
    int64_t getNumLayers() { return num_layers_; }
    double  getDropout() { return dropout_; }
    int     getTrain() { return train_; }
    int     getBidirectional() { return bidirectional_; }
    int     getBatchFirst() { return batch_first_; }
    bool    getMatchCustomOpt() { return match_custom_opt_; }
    int     getCustomOptNumber() { return custom_opt_number_; }

    std::vector<Shape4D> getPreprocessedWeightBlobDim() const override {
       std::vector<Shape4D> weight_blob_shapes;
       for (auto weight_blob_id : weight_blob_ids_) {
          auto weight_blob_shape = getGraph().getBlob(weight_blob_id)->getShape();
          weight_blob_shapes.push_back(weight_blob_shape);
       }
       return weight_blob_shapes;
    }

 private:
    int     has_biases_;
    int64_t num_layers_;
    double  dropout_;
    int     train_;
    int     bidirectional_;
    int     batch_first_;

    std::vector<int64_t> weight_blob_ids_;
    std::vector<int64_t> bias_blob_ids_;

    bool match_custom_opt_;
    int  custom_opt_number_;
}; // class AtenLSTM1Node

} // namespace nn_ir
} // namespace nn_compiler
