#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

//Tensor[] = aten::_set_item(const Tensor& self, int indices, Tensor v)
class AtenSetItemLayer : public NNLayer {
 public:
    /**
     * @brief AtenSetItemLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenSetItemLayer(std::string name, std::string type) : NNLayer(name, type) {
    }

    explicit AtenSetItemLayer(const AtenSetItemLayer& aten_set_item_layer) :
        NNLayer(aten_set_item_layer) {
        this->_indices = aten_set_item_layer._indices;
    }

    virtual ~AtenSetItemLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSetItemLayer>(new AtenSetItemLayer(*this));
    }

    void setIndices(int indices) { _indices = indices; }

    int getIndices() const { return _indices; }

    void printAttr() {
        Log::IR::I() << "    AtenSetItemAttr     ";
        Log::IR::I() << "    indices is          " << _indices;
    }

 private:
    int _indices = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
