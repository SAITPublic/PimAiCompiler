/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenLayerNormLayer : public NNLayer
{
   public:
    AtenLayerNormLayer() {}

    AtenLayerNormLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenLayerNormLayer(const AtenLayerNormLayer& aten_layer_norm_layer) : NNLayer(aten_layer_norm_layer)
    {
        this->normalized_shape_ = aten_layer_norm_layer.normalized_shape_;
    }

    virtual ~AtenLayerNormLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenLayerNormLayer>(new AtenLayerNormLayer(*this));
    }

    void setNormalizedShape(std::vector<int64_t> normalized_shape) { normalized_shape_ = normalized_shape; }

    std::vector<int64_t> getNormalizedShape() const { return normalized_shape_; }

    void setEps(double eps) { eps_ = eps; }

    double getEps() const { return eps_; }

    void setCudnnEnable(int cudnn_enable) { cudnn_enable_ = cudnn_enable; }

    int getCudnnEnable() const { return cudnn_enable_; }

    void setWeights(const std::vector<at::Tensor>& weights) { weights_ = weights; }

    std::vector<at::Tensor> getWeights() { return this->weights_; }

    void setBiases(const std::vector<at::Tensor>& biases) { biases_ = biases; }

    std::vector<at::Tensor> getBiases() { return this->biases_; }

    void setWeightIds(const std::vector<int64_t>& weight_ids) { weight_ids_ = weight_ids; }

    std::vector<int64_t> getWeightIds() { return weight_ids_; }

    void setBiasIds(const std::vector<int64_t>& bias_ids) { bias_ids_ = bias_ids; }

    std::vector<int64_t> getBiasIds() { return bias_ids_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenLayerNormAttr        ";
        DLOG(INFO) << "    normalized_shape is      " << normalized_shape_;
        DLOG(INFO) << "    eps is                   " << eps_;
        DLOG(INFO) << "    cudnn_enable is          " << cudnn_enable_;
    }

   private:
    std::vector<int64_t> normalized_shape_;
    double eps_ = DBL_MAX;
    int cudnn_enable_ = INT32_MAX;

    std::vector<at::Tensor> weights_;
    std::vector<at::Tensor> biases_;
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;
};

}  // namespace ir
}  // namespace nn_compiler
