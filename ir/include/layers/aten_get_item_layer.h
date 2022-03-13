#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenGetItemLayer : public NNLayer {
 public:
    /**
     * @brief AtenGetItemLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenGetItemLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenGetItemLayer(const AtenGetItemLayer& aten_get_item_layer) :
        NNLayer(aten_get_item_layer) {
        this->idx_ = aten_get_item_layer.idx_;
    }

    virtual ~AtenGetItemLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenGetItemLayer>(new AtenGetItemLayer(*this));
    }

    void setIdx(int idx) { idx_ = idx; }

    int getIdx() { return idx_; }

    void printAttr() {
        Log::IR::I() << "    AtenGetItemAttr      ";
    }

 private:
    int idx_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
