#pragma once

#include "new_ir/include/layers/nn_layer.h"
#include "new_ir/include/tensors/data_tensor.h"

namespace nn_compiler {
namespace ir {

class PrimConstantLayer : public NNLayer {
 public:
    /**
     * @brief PrimConstantLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */

    PrimConstantLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit PrimConstantLayer(const PrimConstantLayer& constant_layer) :
        NNLayer(constant_layer) {
        value_ = constant_layer.value_;
    }

    virtual ~PrimConstantLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<PrimConstantLayer>(new PrimConstantLayer(*this));
    }

    void printAttr() {
        Log::IR::I() << "      PrimConstantAttr     ";
    }

    void setAttr(std::shared_ptr<DTensor>  data) {
        value_ = *data;
    }

    std::shared_ptr<DTensor> getAttr() {
        return value_.clone();
    }

    void setNType(std::string& ntype) {
        ntype_ = ntype;
    }

    std::string getNType() const {
        return ntype_;
    }

    void setToRemove(bool to_remove) {
        to_remove_ = to_remove;
    }

    bool getToRemove() {
        return to_remove_;
    }

 private:
    // type : str/device/int/bool/float/Tensor/None/
    // use DTensor store all the type.
    DTensor value_;
    std::string ntype_;
    bool to_remove_ = false;
};

}  // namespace ir
}  // namespace nn_compiler
