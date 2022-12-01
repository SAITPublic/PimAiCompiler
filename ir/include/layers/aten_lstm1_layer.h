/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#pragma once
#include <torch/script.h>
#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

/**
 * API_1:
 * TORCH_API std::tuple<Tensor,Tensor,Tensor> lstm(
 *  const Tensor & input,
 *  TensorList hx,
 *  TensorList params,
 *  bool has_biases,
 *  int64_t num_layers,
 *  double dropout,
 *  bool train,
 *  bool bidirectional,
 *  bool batch_first)
 *
 */

namespace nn_compiler
{
namespace ir
{
class AtenLSTM1Layer : public NNLayer
{
   public:
    /**
     * @brief Construct a new Aten LSTM1 Layer object
     *
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenLSTM1Layer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenLSTM1Layer(const AtenLSTM1Layer &aten_lstm1_layer) : NNLayer(aten_lstm1_layer)
    {
        this->weights_ = aten_lstm1_layer.weights_;
        this->biases_ = aten_lstm1_layer.biases_;
        this->weight_ids_ = aten_lstm1_layer.weight_ids_;
        this->bias_ids_ = aten_lstm1_layer.bias_ids_;
        this->param_vector_ = aten_lstm1_layer.param_vector_;
        this->arranged_weight_ = aten_lstm1_layer.arranged_weight_;
        this->setAttr(aten_lstm1_layer.has_biases_, aten_lstm1_layer.num_layers_, aten_lstm1_layer.dropout_,
                      aten_lstm1_layer.train_, aten_lstm1_layer.bidirectional_, aten_lstm1_layer.batch_first_,
                      aten_lstm1_layer.match_custom_opt_, aten_lstm1_layer.custom_opt_number_);
    }

    virtual ~AtenLSTM1Layer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenLSTM1Layer>(new AtenLSTM1Layer(*this)); }

    void printAttr()
    {
        DLOG(INFO) << "     AtenLSTM1Attr   ";
        DLOG(INFO) << "     has_biases is        " << this->has_biases_;
        DLOG(INFO) << "     num_layers is        " << this->num_layers_;
        DLOG(INFO) << "     dropout is           " << this->dropout_;
        DLOG(INFO) << "     bidirectional is     " << this->bidirectional_;
        DLOG(INFO) << "     train is             " << this->train_;
        DLOG(INFO) << "     batch_first is       " << this->batch_first_;
        DLOG(INFO) << "     match_custom_opt is  " << this->match_custom_opt_;
        DLOG(INFO) << "     custom_opt_number is " << this->custom_opt_number_;
    }

    void setAttr(int has_biases, int64_t num_layers, double dropout, int train, int bidirectional, int batch_first,
                 bool match_custom_opt, int custom_opt_number)
    {
        this->has_biases_ = has_biases;
        this->num_layers_ = num_layers;
        this->dropout_ = dropout;
        this->train_ = train;
        this->bidirectional_ = bidirectional;
        this->batch_first_ = batch_first;
        this->match_custom_opt_ = match_custom_opt;
        this->custom_opt_number_ = custom_opt_number;
    }

    void setWeights(const std::vector<at::Tensor> &weights) { this->weights_ = weights; }

    std::vector<at::Tensor> getWeights() { return this->weights_; }

    void setBiases(const std::vector<at::Tensor> &biases) { this->biases_ = biases; }

    std::vector<at::Tensor> getBiases() { return this->biases_; }

    void setWeightIds(const std::vector<int64_t> &weight_ids) { weight_ids_ = weight_ids; }

    std::vector<int64_t> getWeightIds() { return weight_ids_; }

    void setBiasIds(const std::vector<int64_t> &bias_ids) { bias_ids_ = bias_ids; }

    std::vector<int64_t> getBiasIds() { return bias_ids_; }

    void setParamVector(const std::vector<at::Tensor> &param_vector) { this->param_vector_ = param_vector; }

    std::vector<at::Tensor> getParamVector() { return this->param_vector_; }

    void setArrangedWeight(const at::Tensor &arranged_weight) { this->arranged_weight_ = arranged_weight; }

    at::Tensor getArrangedWeight() { return this->arranged_weight_; }

    void setHasBiases(int has_biases) { this->has_biases_ = has_biases; }

    int getHasBiases() { return this->has_biases_; }

    void setNumLayers(int64_t num_layers) { this->num_layers_ = num_layers; }

    int64_t getNumLayers() { return this->num_layers_; }

    void setDropout(double dropout) { this->dropout_ = dropout; }

    double getDropout() { return dropout_; }

    void setTrain(int train) { this->train_ = train; }

    int getTrain() { return this->train_; }

    void setBidirectional(int bidirectional) { this->bidirectional_ = bidirectional; }

    int getBidirectional() { return this->bidirectional_; }

    void setBatchFirst(int batch_first) { this->batch_first_ = batch_first; }

    int getBatchFirst() { return this->batch_first_; }

    void setMatchCustomOpt(bool match_custom_opt) { match_custom_opt_ = match_custom_opt; }

    bool getMatchCustomOpt() { return match_custom_opt_; }

    void setCustomOptNumber(int custom_opt_number) { custom_opt_number_ = custom_opt_number; }

    int getCustomOptNumber() { return custom_opt_number_; }

    void setCustomCatMemId(int custom_cat_mem_id) { custom_cat_mem_id_ = custom_cat_mem_id; }

    int getCustomCatMemId() { return custom_cat_mem_id_; }

    struct AtenLSTM1LayerAttr {
        int has_biases;
        int64_t num_layers;
        double dropout;
        int train;
        int bidirectional;
        int batch_first;
        bool match_custom_opt;
        int custom_opt_number;
    };

    AtenLSTM1LayerAttr getAttr()
    {
        AtenLSTM1LayerAttr attrs;
        attrs.has_biases = this->has_biases_;
        attrs.num_layers = this->num_layers_;
        attrs.dropout = this->dropout_;
        attrs.train = this->train_;
        attrs.bidirectional = this->bidirectional_;
        attrs.batch_first = this->batch_first_;
        attrs.match_custom_opt = this->match_custom_opt_;
        attrs.custom_opt_number = this->custom_opt_number_;
        return attrs;
    }

   private:
    int has_biases_ = INT32_MAX;
    int64_t num_layers_ = INT64_MIN;
    double dropout_ = DBL_MAX;
    int train_ = INT32_MAX;
    int bidirectional_ = INT32_MAX;
    int batch_first_ = INT32_MAX;

    // weights & bias, 8 or 12 tensors
    std::vector<at::Tensor> weights_;  // only weight, dim > 1
    std::vector<at::Tensor> biases_;   // bias, dim == 1
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;

    std::vector<at::Tensor> param_vector_;  // weight and bias
    at::Tensor arranged_weight_;

    int lstm_type_ = 0;
    bool match_custom_opt_ = false;
    int custom_opt_number_ = 0;
    int custom_cat_mem_id_ = 0;
};
}  // namespace ir
}  // namespace nn_compiler
