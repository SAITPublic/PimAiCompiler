#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenSqueezeLayer : public NNLayer {
 public:
    AtenSqueezeLayer() {}

    AtenSqueezeLayer(std::string name, LayerType type) : NNLayer(name, type) {
    }

    explicit AtenSqueezeLayer(const AtenSqueezeLayer& aten_squeeze_layer) :
        NNLayer(aten_squeeze_layer) {
        this->_dim = aten_squeeze_layer._dim;
    }

    virtual ~AtenSqueezeLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSqueezeLayer>(new AtenSqueezeLayer(*this));
    }

    void setDim(int64_t dim) { _dim = dim; }

    int64_t getDim() const { return _dim; }

    void printAttr() {
        Log::IR::I() << "    AtenSqueezeAttr    ";
        Log::IR::I() << "    dim is              " << _dim;
    }

 private:
    int64_t _dim = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
