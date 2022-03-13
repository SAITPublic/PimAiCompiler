#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenMaskedFillLayer : public NNLayer {
 public:
    AtenMaskedFillLayer() {}

    AtenMaskedFillLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenMaskedFillLayer(const AtenMaskedFillLayer &aten_masked_fill_layer) :
        NNLayer(aten_masked_fill_layer) {
    }

    virtual ~AtenMaskedFillLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenMaskedFillLayer>(new AtenMaskedFillLayer(*this));
    }

    void setIsInplace(bool is_inplace) { is_inplace_ = is_inplace; }

    bool getIsInplace() const { return is_inplace_; }

    void printAttr() {
        Log::IR::I() << "    AtenMaskedFillAttr      ";
    }

 private:
    bool is_inplace_ = false;
};

} // namespace ir
} // namespace nn_compiler
