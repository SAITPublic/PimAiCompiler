
#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenIndexSelectLayer : public NNLayer {
 public:
    AtenIndexSelectLayer() {}

    AtenIndexSelectLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenIndexSelectLayer(const AtenIndexSelectLayer& aten_index_select_layer) :
        NNLayer(aten_index_select_layer) {
        this->_dim = aten_index_select_layer._dim;
    }

    virtual ~AtenIndexSelectLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenIndexSelectLayer>(new AtenIndexSelectLayer(*this));
    }

    void setDim(int dim) { _dim = dim; }

    int getDim() const { return _dim; }

    void printAttr() {
        Log::IR::I() << "    AtenIndexSelectAttr     ";
        Log::IR::I() << "    dim is                  "<< _dim;
    }

 private:
    int _dim = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
