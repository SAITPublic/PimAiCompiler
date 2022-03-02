#pragma once

#include "new_ir/include/tensors/data_tensor.h"
#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class PrimGetAttrLayer : public NNLayer {
 public:
    /**
     * @brief PrimGetAttrLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    PrimGetAttrLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit PrimGetAttrLayer(const PrimGetAttrLayer& get_attr_layer) :
        NNLayer(get_attr_layer) {
        this->values_ = get_attr_layer.values_;
        this->ntype_  = get_attr_layer.ntype_;
    }

    virtual ~PrimGetAttrLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<PrimGetAttrLayer>(new PrimGetAttrLayer(*this));
    }

    void printAttr() {
        Log::IR::I() << "      PrimGetAttrAttr     ";
    }

 private:
    std::vector<std::shared_ptr<DTensor>> values_;
    std::string ntype_;
};

}  // namespace ir
}  // namespace nn_compiler
