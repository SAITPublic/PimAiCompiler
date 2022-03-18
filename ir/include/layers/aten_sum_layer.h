#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//  sum(const Tensor& self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype)
class AtenSumLayer : public NNLayer {
 public:
    AtenSumLayer() {}

    AtenSumLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenSumLayer(const AtenSumLayer& aten_sum_layer) : NNLayer(aten_sum_layer) {
        this->dim_ = aten_sum_layer.dim_;
        this->keepdim_ = aten_sum_layer.keepdim_;
        this->dtype_ = aten_sum_layer.dtype_;
    }

    virtual ~AtenSumLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSumLayer>(new AtenSumLayer(*this));
    }

    void setDim(const std::vector<int64_t> &dim) { dim_ = dim; }

    const std::vector<int64_t> getDim() const { return dim_; }

    void setKeepdim(int keepdim) { keepdim_ = keepdim; }

    int getKeepdim() const { return keepdim_; }

    void setDtype(int64_t dtype) { dtype_ = dtype; }

    int64_t getDtype() const { return dtype_; }

    void printAttr() {
        DLOG(INFO) << "    AtenSumAttr         ";
        DLOG(INFO) << "    dim is              " << &dim_;
        DLOG(INFO) << "    keepdim is          " << keepdim_;
        DLOG(INFO) << "    dtype is            " << dtype_;
    }

 private:
    std::vector<int64_t> dim_;
    int keepdim_   = INT32_MAX;
    int64_t dtype_ = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
