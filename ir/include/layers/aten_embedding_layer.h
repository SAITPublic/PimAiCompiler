#pragma once

#include <torch/script.h>
#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenEmbeddingLayer : public NNLayer {
 public:
    AtenEmbeddingLayer() {}

    AtenEmbeddingLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenEmbeddingLayer(const AtenEmbeddingLayer &aten_embedding_layer) :
        NNLayer(aten_embedding_layer) {
        this->_weights = aten_embedding_layer._weights;
        this->_weights_shape = aten_embedding_layer._weights_shape;
        this->_padding_idx = aten_embedding_layer._padding_idx;
        this->_sparse = aten_embedding_layer._sparse;
        this->_scale_grad_by_freq = aten_embedding_layer._scale_grad_by_freq;
    }

    virtual ~AtenEmbeddingLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenEmbeddingLayer>(new AtenEmbeddingLayer(*this));
    }

    void setWeights(std::vector<at::Tensor> weights) { _weights = weights; }

    std::vector<at::Tensor> getWeights() { return _weights; }

    void setWeightsShape(std::vector<int> weights_shape) { _weights_shape = weights_shape; }

    std::vector<int> getWeightsShape() { return _weights_shape; }

    void setPaddingIdx(int64_t padding_idx) { _padding_idx = padding_idx; }

    int64_t getPaddingIdx() const { return _padding_idx; }

    void setScaleGrad(int scale_grad_by_freq) { _scale_grad_by_freq = scale_grad_by_freq; }

    int getScaleGrad() const { return _scale_grad_by_freq; }

    void setSparse(int sparse) { _sparse = sparse; }

    int getSparse() const { return _sparse; }

    void printAttr() {
        DLOG(INFO) << "    AtenEmbeddingAttr      ";
        DLOG(INFO) << "    padding_idx is         "<< _padding_idx;
        DLOG(INFO) << "    scale_grad_by_freq is  "<< _scale_grad_by_freq;
        DLOG(INFO) << "    sparse is              "<< _sparse;
    }

 private:
    std::vector<at::Tensor> _weights;
    std::vector<int> _weights_shape;
    int64_t  _padding_idx       = INT64_MIN;
    int     _scale_grad_by_freq = INT32_MAX;
    int     _sparse             = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
