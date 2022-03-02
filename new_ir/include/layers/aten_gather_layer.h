#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenGatherLayer : public NNLayer {
 public:
    AtenGatherLayer() {}

    AtenGatherLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenGatherLayer(const AtenGatherLayer &aten_gather_layer) :
        NNLayer(aten_gather_layer) {
        this->_dim = aten_gather_layer._dim;
        this->_sparse_grad = aten_gather_layer._sparse_grad;
    }

    virtual ~AtenGatherLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenGatherLayer>(new AtenGatherLayer(*this));
    }

    void setDim(int dim) { _dim = dim; }

    int getDim() const { return _dim; }

    void setSparseGrad(int sparse_grad) { _sparse_grad = sparse_grad; }

    int getSparseGrad() const { return _sparse_grad; }

    void printAttr() {
        Log::IR::I() << "    AtenGatherAttr      ";
        Log::IR::I() << "    dim is              "<< _dim;
        Log::IR::I() << "    sparse_grad is      "<< _sparse_grad;
    }

 private:
    int  _dim        = INT32_MAX;
    int _sparse_grad = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
