#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenAddLayer : public NNLayer {
 public:
    /**
     * @brief AtenAddLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenAddLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenAddLayer(const AtenAddLayer& aten_add_layer) : NNLayer(aten_add_layer) {
        this->alpha_ = aten_add_layer.alpha_;
    }

    virtual ~AtenAddLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenAddLayer>(new AtenAddLayer(*this));
    }

    void printAttr() {
        Log::IR::I() << "    AtenAddAttr      ";
    }

    void setAlpha(int64_t alpha) { alpha_ = alpha; }

    int64_t getAlpha() { return alpha_; }

 private:
    int64_t alpha_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
