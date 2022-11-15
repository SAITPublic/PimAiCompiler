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

#include <torch/script.h>
#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenEmbeddingLayer : public NNLayer
{
   public:
    AtenEmbeddingLayer() {}

    AtenEmbeddingLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenEmbeddingLayer(const AtenEmbeddingLayer &aten_embedding_layer) : NNLayer(aten_embedding_layer)
    {
        this->weights_ = aten_embedding_layer.weights_;
        this->weights_shape_ = aten_embedding_layer.weights_shape_;
        this->padding_idx_ = aten_embedding_layer.padding_idx_;
        this->sparse_ = aten_embedding_layer.sparse_;
        this->scale_grad_by_freq_ = aten_embedding_layer.scale_grad_by_freq_;
    }

    virtual ~AtenEmbeddingLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenEmbeddingLayer>(new AtenEmbeddingLayer(*this));
    }

    void setWeights(std::vector<at::Tensor> weights) { weights_ = weights; }

    std::vector<at::Tensor> getWeights() { return weights_; }

    void setWeightsShape(std::vector<int> weights_shape) { weights_shape_ = weights_shape; }

    std::vector<int> getWeightsShape() { return weights_shape_; }

    void setPaddingIdx(int64_t padding_idx) { padding_idx_ = padding_idx; }

    int64_t getPaddingIdx() const { return padding_idx_; }

    void setScaleGrad(int scale_grad_by_freq) { scale_grad_by_freq_ = scale_grad_by_freq; }

    int getScaleGrad() const { return scale_grad_by_freq_; }

    void setSparse(int sparse) { sparse_ = sparse; }

    int getSparse() const { return sparse_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenEmbeddingAttr      ";
        DLOG(INFO) << "    padding_idx is         " << padding_idx_;
        DLOG(INFO) << "    scale_grad_by_freq is  " << scale_grad_by_freq_;
        DLOG(INFO) << "    sparse is              " << sparse_;
    }

   private:
    std::vector<at::Tensor> weights_;
    std::vector<int> weights_shape_;
    int64_t padding_idx_ = INT64_MIN;
    int scale_grad_by_freq_ = INT32_MAX;
    int sparse_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
