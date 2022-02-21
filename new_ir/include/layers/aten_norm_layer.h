
#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenNormLayer : public NNLayer {
 public:
    AtenNormLayer() {}

    AtenNormLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenNormLayer(const AtenNormLayer& aten_norm_layer) :
        NNLayer(aten_norm_layer) {
        this->p_ = aten_norm_layer.p_;
    }

    virtual ~AtenNormLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenNormLayer>(new AtenNormLayer(*this));
    }

    void setP(int64_t p) { p_ = p; }

    int64_t getP() { return p_; }

    void printAttr() {
        Log::IR::I() << "    AtenNormAttr      ";
    }

 private:
    int64_t p_ = INT64_MIN;
};

} // namespace ir
} // namespace nn_compiler
