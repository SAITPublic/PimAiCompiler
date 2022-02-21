#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenExpandLayer : public NNLayer {
 public:
    AtenExpandLayer() {}

    AtenExpandLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenExpandLayer(const AtenExpandLayer &aten_expand_layer) :
        NNLayer(aten_expand_layer) {
        this->_implicit = aten_expand_layer._implicit;
    }

    virtual ~AtenExpandLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenExpandLayer>(new AtenExpandLayer(*this));
    }

    void setImplicit(int implicit) { _implicit = implicit; }

    int getImplicit() { return _implicit; }


    void printAttr() {
        Log::IR::I() << "    AtenExpandAttr      ";
        Log::IR::I() << "    implicit is         " << _implicit;
    }

 private:
    int _implicit = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
