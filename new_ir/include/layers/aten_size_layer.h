#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//def size(g, self, dim=None)
class AtenSizeLayer : public NNLayer {
 public:
    AtenSizeLayer() {}

    AtenSizeLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenSizeLayer(const AtenSizeLayer& aten_size_layer) :
        NNLayer(aten_size_layer) {
        this->_dim = aten_size_layer.getDim();
    }

    virtual ~AtenSizeLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSizeLayer>(new AtenSizeLayer(*this));
    }

    void setDim(int64_t dim) { _dim = dim; }

    int64_t getDim() const { return _dim; }

    void printAttr() {
        Log::IR::I() << "    AtenSizeAttr      ";
        Log::IR::I() << "    dim is           " << _dim;
    }

 private:
    int64_t _dim = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
