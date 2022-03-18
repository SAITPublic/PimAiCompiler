#pragma once

#include "ir/include/layers/nn_layer.h"

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
    AtenSetItemLayer(std::string name, LayerType type) : NNLayer(name, type) {
    }

    explicit AtenSetItemLayer(const AtenSetItemLayer& aten_set_item_layer) :
        NNLayer(aten_set_item_layer) {
        this->indices_ = aten_set_item_layer.indices_;
    }

    virtual ~AtenSetItemLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenSetItemLayer>(new AtenSetItemLayer(*this));
    }

    void setIndices(int indices) { indices_ = indices; }

    int getIndices() const { return indices_; }

    void printAttr() {
        DLOG(INFO) << "    AtenSetItemAttr     ";
        DLOG(INFO) << "    indices is          " << indices_;
    }

 private:
    int indices_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
