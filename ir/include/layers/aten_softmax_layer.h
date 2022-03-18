
#pragma once

#include "ir/include/layers/nn_layer.h"

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
        this->dim_ = aten_softmax_layer.dim_;
        this->dtype_ = aten_softmax_layer.dtype_;
    }

    virtual ~AtenSoftmaxLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSoftmaxLayer>(new AtenSoftmaxLayer(*this));
    }

    void setDim(int64_t dim) { dim_ = dim; }

    void setDtype(int64_t dtype) { dtype_ = dtype; }

    int64_t getDim() const { return dim_; }

    int64_t getDtype() const { return dtype_; }

    void printAttr() {
        DLOG(INFO) << "    AtenSoftmaxAttr";
        DLOG(INFO) << "    dim is         " << dim_;
        DLOG(INFO) << "    dtype is       " << dtype_;
    }

 private:
    int64_t dim_   = INT64_MIN;
    int64_t dtype_ = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
