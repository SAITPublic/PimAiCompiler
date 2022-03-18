#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/tensors/data_tensor.h"

namespace nn_compiler {
namespace ir {

// aten::warn[warn_id=0](str, int)
class AtenWarnLayer : public NNLayer {
 public:
    /**
     * @brief AtenWarnLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    AtenWarnLayer(std::string name, LayerType type) : NNLayer(name, type) {
    }

    explicit AtenWarnLayer(const AtenWarnLayer& aten_warn_layer) :
        NNLayer(aten_warn_layer) {
        this->_value = aten_warn_layer._value;
    }

    virtual ~AtenWarnLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenWarnLayer>(new AtenWarnLayer(*this));
    }

    void setValue(int value) { _value = value; }

    int getValue() const { return _value; }

    void printAttr() {
        DLOG(INFO) << "      AtenWarnAttr     ";
        DLOG(INFO) << "      value is         "<< _value;
    }

 private:
    int _value = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
