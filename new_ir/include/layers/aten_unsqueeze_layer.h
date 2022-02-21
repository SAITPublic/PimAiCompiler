#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenUnsqueezeLayer : public NNLayer {
 public:
    /**
     * @brief AtenUnsqueezeLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenUnsqueezeLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenUnsqueezeLayer(const AtenUnsqueezeLayer& aten_unsqueeze_layer) :
        NNLayer(aten_unsqueeze_layer) {
        this->_dim = aten_unsqueeze_layer._dim;
    }

    virtual ~AtenUnsqueezeLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenUnsqueezeLayer>(new AtenUnsqueezeLayer(*this));
    }

    void setDim(int64_t dim) { _dim = dim; }

    void setIsInplace(bool is_inplace) { _is_inplace = is_inplace; }

    int64_t getDim() const { return _dim; }

    bool getIsInplace() const { return _is_inplace; }

    void printAttr() {
        Log::IR::I() << "     AtenUnsqueezeAttr      ";
        Log::IR::I() << "     dim is                 "<< _dim;
    }

 private:
    int64_t _dim = INT64_MIN;
    bool _is_inplace = false;
};

}  // namespace ir
}  // namespace nn_compiler
