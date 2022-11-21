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
// Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias,
//               IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
//               int64_t groups);
class AtenConv2dLayer : public NNLayer
{
   public:
    /**
     * @brief AtenConv2dLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenConv2dLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenConv2dLayer(const AtenConv2dLayer &aten_conv2d_layer) : NNLayer(aten_conv2d_layer)
    {
        this->weights_ = aten_conv2d_layer.weights_;
        this->bias_ = aten_conv2d_layer.bias_;
        this->weight_ids_ = aten_conv2d_layer.weight_ids_;
        this->bias_ids_ = aten_conv2d_layer.bias_ids_;
        this->stride_ = aten_conv2d_layer.stride_;
        this->padding_ = aten_conv2d_layer.padding_;
        this->dialation_ = aten_conv2d_layer.dialation_;
        this->groups_ = aten_conv2d_layer.groups_;
    }

    virtual ~AtenConv2dLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenConv2dLayer>(new AtenConv2dLayer(*this)); }

    void setWeights(const std::vector<at::Tensor> &weights) { weights_ = weights; }

    void setBiases(const std::vector<at::Tensor> &bias) { bias_ = bias; }

    void setWeightIds(const std::vector<int64_t> &weight_ids) { weight_ids_ = weight_ids; }

    void setBiasIds(const std::vector<int64_t> &bias_ids) { bias_ids_ = bias_ids; }

    void setStride(const std::vector<int64_t> &stride) { stride_ = stride; }

    void setPadding(const std::vector<int64_t> &padding) { padding_ = padding; }

    void setDialation(const std::vector<int64_t> &dialation) { dialation_ = dialation; }

    void setGroups(int64_t groups) { groups_ = groups; }

    std::vector<at::Tensor> getWeights() { return this->weights_; }

    std::vector<at::Tensor> getBiases() { return this->bias_; }

    std::vector<int64_t> getWeightIds() { return weight_ids_; }

    std::vector<int64_t> getBiasIds() { return bias_ids_; }

    const std::vector<int64_t> getStride() const { return stride_; }

    const std::vector<int64_t> getPadding() const { return padding_; }

    const std::vector<int64_t> getDialation() const { return dialation_; }

    int64_t getGroups() const { return groups_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenConv2dAttr          ";
        DLOG(INFO) << "    stride[0] is            " << stride_[0];
        DLOG(INFO) << "    stride[1] is            " << stride_[1];
        DLOG(INFO) << "    padding[0] is           " << padding_[0];
        DLOG(INFO) << "    padding[1] is           " << padding_[1];
        DLOG(INFO) << "    dialation[0] is         " << dialation_[0];
        DLOG(INFO) << "    dialation[1] is         " << dialation_[1];
        DLOG(INFO) << "    groups is               " << groups_;
    }

   private:
    std::vector<at::Tensor> weights_;
    std::vector<at::Tensor> bias_;
    std::vector<int64_t> weight_ids_;
    std::vector<int64_t> bias_ids_;
    std::vector<int64_t> stride_ = {1, 1};
    std::vector<int64_t> padding_ = {0, 0};
    std::vector<int64_t> dialation_ = {1, 1};

    int64_t groups_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
