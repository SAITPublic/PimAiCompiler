#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//  sum(const Tensor& self, DimnameList dim, bool keepdim, c10::optional<ScalarType> dtype)
class AtenSumLayer : public NNLayer {
 public:
    AtenSumLayer() {}

    AtenSumLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenSumLayer(const AtenSumLayer& aten_sum_layer) : NNLayer(aten_sum_layer) {
        this->_dim = aten_sum_layer._dim;
        this->_keepdim = aten_sum_layer._keepdim;
        this->_dtype = aten_sum_layer._dtype;
    }

    virtual ~AtenSumLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSumLayer>(new AtenSumLayer(*this));
    }

    void setDim(const std::vector<int64_t> &dim) { _dim = dim; }

    const std::vector<int64_t> getDim() const { return _dim; }

    void setKeepdim(int keepdim) { _keepdim = keepdim; }

    int getKeepdim() const { return _keepdim; }

    void setDtype(int64_t dtype) { _dtype = dtype; }

    int64_t getDtype() const { return _dtype; }

    void printAttr() {
        Log::IR::I() << "    AtenSumAttr         ";
        Log::IR::I() << "    dim is              " << &_dim;
        Log::IR::I() << "    keepdim is          " << _keepdim;
        Log::IR::I() << "    dtype is            " << _dtype;
    }

 private:
    std::vector<int64_t> _dim;
    int _keepdim   = INT32_MAX;
    int64_t _dtype = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
