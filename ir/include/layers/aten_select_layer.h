#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//Tensor select(const Tensor& self, int64_t dim, int64_t index)
class AtenSelectLayer : public NNLayer {
 public:
    /**
     * @brief AtenSelectLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenSelectLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenSelectLayer(const AtenSelectLayer& aten_select_layer) :
        NNLayer(aten_select_layer) {
        this->_dim = aten_select_layer._dim;
        this->_index = aten_select_layer._index;
    }

    virtual ~AtenSelectLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSelectLayer>(new AtenSelectLayer(*this));
    }

    void setDim(int64_t dim) { _dim = dim; }

    int64_t getDim() const { return _dim; }

    void setIndex(int64_t index) { _index = index; }

    int64_t getIndex() const { return _index; }

    void printAttr() {
        Log::IR::I() << "    AtenSelectAttr     ";
        Log::IR::I() << "    dim is             " << _dim;
        Log::IR::I() << "    index is           " << _index;
    }

 private:
    int64_t _dim   = INT64_MIN;
    int64_t _index = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
