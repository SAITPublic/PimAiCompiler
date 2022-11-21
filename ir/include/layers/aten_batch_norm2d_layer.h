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

namespace nn_compiler
{
namespace ir
{
class AtenBatchNorm2dLayer : public NNLayer
{
   public:
    AtenBatchNorm2dLayer() {}

    AtenBatchNorm2dLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenBatchNorm2dLayer(const AtenBatchNorm2dLayer& batch_norm2d_layer) : NNLayer(batch_norm2d_layer)
    {
        this->weights_ = batch_norm2d_layer.weights_;
        this->bias_ = batch_norm2d_layer.bias_;
        this->weight_ids_ = batch_norm2d_layer.weight_ids_;
        this->bias_ids_ = batch_norm2d_layer.bias_ids_;
        this->training_ = batch_norm2d_layer.training_;
        this->momentum_ = batch_norm2d_layer.momentum_;
        this->eps_ = batch_norm2d_layer.eps_;
        this->cudnn_enabled_ = batch_norm2d_layer.cudnn_enabled_;
    }

    virtual ~AtenBatchNorm2dLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenBatchNorm2dLayer>(new AtenBatchNorm2dLayer(*this));
    }

    std::vector<at::Tensor> getWeights() { return this->weights_; }

    std::vector<at::Tensor> getBiases() { return this->bias_; }

    std::vector<int64_t> getWeightIds() { return weight_ids_; }

    std::vector<int64_t> getBiasIds() { return bias_ids_; }

    int getTraining() { return training_; }

    double getMomentum() { return momentum_; }

    double getEps() { return eps_; }

    int getCudnnEnabled() { return cudnn_enabled_; }

    void setWeights(const std::vector<at::Tensor>& weights) { weights_ = weights; }

    void setBiases(const std::vector<at::Tensor>& bias) { bias_ = bias; }

    void setWeightIds(const std::vector<int64_t>& weight_ids) { weight_ids_ = weight_ids; }

    void setBiasIds(const std::vector<int64_t>& bias_ids) { bias_ids_ = bias_ids; }

    void setTraining(int training) { training_ = training; }

    void setMomentum(double momentum) { momentum_ = momentum; }

    void setEps(double eps) { eps_ = eps; }

    void setCudnnEnabled(int cudnn_enabled) { cudnn_enabled_ = cudnn_enabled; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenAsTensorAttr       ";
        DLOG(INFO) << "    training is          " << training_;
        DLOG(INFO) << "    momentum is          " << momentum_;
        DLOG(INFO) << "    eps is               " << eps_;
        DLOG(INFO) << "    cudnn_enabled is     " << cudnn_enabled_;
    }

   private:
    std::vector<at::Tensor> weights_;
    std::vector<at::Tensor> bias_;
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;
    int training_ = INT32_MAX;
    double momentum_ = DBL_MAX;
    double eps_ = DBL_MAX;
    int cudnn_enabled_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
