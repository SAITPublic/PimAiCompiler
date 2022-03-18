#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenExpandLayer : public NNLayer {
 public:
    AtenExpandLayer() {}

    AtenExpandLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenExpandLayer(const AtenExpandLayer &aten_expand_layer) :
        NNLayer(aten_expand_layer) {
        this->implicit_ = aten_expand_layer.implicit_;
    }

    virtual ~AtenExpandLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenExpandLayer>(new AtenExpandLayer(*this));
    }

    void setImplicit(int implicit) { implicit_ = implicit; }

    int getImplicit() { return implicit_; }


    void printAttr() {
        DLOG(INFO) << "    AtenExpandAttr      ";
        DLOG(INFO) << "    implicit is         " << implicit_;
    }

 private:
    int implicit_ = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
