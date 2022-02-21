#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenEmbeddinNNLayer : public NNLayer {
 public:
    AtenEmbeddinNNLayer() {}

    AtenEmbeddinNNLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenEmbeddinNNLayer(const AtenEmbeddinNNLayer &aten_embedding_layer) :
        NNLayer(aten_embedding_layer) {
        this->_weights = aten_embedding_layer._weights;
        this->_weights_shape = aten_embedding_layer._weights_shape;
        this->_padding_idx = aten_embedding_layer._padding_idx;
        this->_sparse = aten_embedding_layer._sparse;
        this->_scale_grad_by_freq = aten_embedding_layer._scale_grad_by_freq;
    }

    virtual ~AtenEmbeddinNNLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenEmbeddinNNLayer>(new AtenEmbeddinNNLayer(*this));
    }

    void setWeights(std::vector<float> weights) { _weights = weights; }

    std::vector<float> getWeights() { return _weights; }

    void setWeightsShape(std::vector<int> weights_shape) { _weights_shape = weights_shape; }

    std::vector<int> getWeightsShape() { return _weights_shape; }

    void setPaddingIdx(int64_t padding_idx) { _padding_idx = padding_idx; }

    int64_t getPaddingIdx() const { return _padding_idx; }

    void setScaleGrad(int scale_grad_by_freq) { _scale_grad_by_freq = scale_grad_by_freq; }

    int getScaleGrad() const { return _scale_grad_by_freq; }

    void setSparse(int sparse) { _sparse = sparse; }

    int getSparse() const { return _sparse; }

    void printAttr() {
        Log::IR::I() << "    AtenEmbeddingAttr      ";
        Log::IR::I() << "    padding_idx is         "<< _padding_idx;
        Log::IR::I() << "    scale_grad_by_freq is  "<< _scale_grad_by_freq;
        Log::IR::I() << "    sparse is              "<< _sparse;
    }

 private:
    std::vector<float> _weights;
    std::vector<int> _weights_shape;
    int64_t  _padding_idx       = INT64_MIN;
    int     _scale_grad_by_freq = INT32_MAX;
    int     _sparse             = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
