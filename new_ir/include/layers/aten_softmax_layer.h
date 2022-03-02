
#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenSoftmaxLayer : public NNLayer {
 public:
    AtenSoftmaxLayer() {}

    AtenSoftmaxLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenSoftmaxLayer(const AtenSoftmaxLayer& aten_softmax_layer) :
        NNLayer(aten_softmax_layer) {
        this->_dim = aten_softmax_layer._dim;
        this->_dtype = aten_softmax_layer._dtype;
    }

    virtual ~AtenSoftmaxLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSoftmaxLayer>(new AtenSoftmaxLayer(*this));
    }

    void setDim(int64_t dim) { _dim = dim; }

    void setDtype(int64_t dtype) { _dtype = dtype; }

    int64_t getDim() const { return _dim; }

    int64_t getDtype() const { return _dtype; }

    void printAttr() {
        Log::IR::I() << "    AtenSoftmaxAttr";
        Log::IR::I() << "    dim is         " << _dim;
        Log::IR::I() << "    dtype is       " << _dtype;
    }

 private:
    int64_t _dim   = INT64_MIN;
    int64_t _dtype = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
