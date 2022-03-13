#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenLogSoftmaxLayer : public NNLayer {
 public:
    AtenLogSoftmaxLayer() {}

    AtenLogSoftmaxLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenLogSoftmaxLayer(const AtenLogSoftmaxLayer &aten_log_softmax_layer) :
        NNLayer(aten_log_softmax_layer) {
        this->_dim = aten_log_softmax_layer._dim;
        this->_dtype = aten_log_softmax_layer._dtype;
    }

    virtual ~AtenLogSoftmaxLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenLogSoftmaxLayer>(new AtenLogSoftmaxLayer(*this));
    }

    void setDim(int64_t dim) { _dim = dim; }

    void setDType(int64_t dtype) { _dtype = dtype; }

    int64_t getDim() const { return _dim; }

    int64_t getDType() const { return _dtype; }

    void printAttr() {
        Log::IR::I() << "    AtenLogSoftmaxAttr      ";
        Log::IR::I() << "    dim is                  " << _dim;
        Log::IR::I() << "    dtype is                " << _dtype;
    }

 private:
    int64_t _dim   = INT64_MIN;
    int64_t _dtype = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
