#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenSqueezeLayer : public NNLayer {
 public:
    AtenSqueezeLayer() {}

    AtenSqueezeLayer(std::string name, LayerType type) : NNLayer(name, type) {
    }

    explicit AtenSqueezeLayer(const AtenSqueezeLayer& aten_squeeze_layer) :
        NNLayer(aten_squeeze_layer) {
        this->dim_ = aten_squeeze_layer.dim_;
    }

    virtual ~AtenSqueezeLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSqueezeLayer>(new AtenSqueezeLayer(*this));
    }

    void setDim(int64_t dim) { dim_ = dim; }

    int64_t getDim() const { return dim_; }

    void printAttr() {
        DLOG(INFO) << "    AtenSqueezeAttr    ";
        DLOG(INFO) << "    dim is              " << dim_;
    }

 private:
    int64_t dim_ = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
