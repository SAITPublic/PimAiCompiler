#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenLeakyReluLayer : public NNLayer {
 public:
    AtenLeakyReluLayer() {}

    AtenLeakyReluLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenLeakyReluLayer(const AtenLeakyReluLayer &aten_leaky_relu_layer) :
        NNLayer(aten_leaky_relu_layer) {
        this->_scalar = aten_leaky_relu_layer._scalar;
    }

    virtual ~AtenLeakyReluLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenLeakyReluLayer>(new AtenLeakyReluLayer(*this));
    }

    void setScalar(double scalar) { _scalar = scalar; }

    double getScalar() const { return _scalar; }

    void printAttr() {
        DLOG(INFO) << "    AtenLeakyReluAttr      ";
        DLOG(INFO) << "    sclar is               " << _scalar;
    }

 private:
    double _scalar = DBL_MAX;
};

} // namespace ir
} // namespace nn_compiler
