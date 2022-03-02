#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenDeriveIndexLayer : public NNLayer {
 public:
    /**
     * @brief AtenDeriveIndexLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    AtenDeriveIndexLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenDeriveIndexLayer(const AtenDeriveIndexLayer& aten_derive_index_layer) :
            NNLayer(aten_derive_index_layer) {
                this->_start = aten_derive_index_layer._start;
                this->_step  = aten_derive_index_layer._step;
            }

    virtual ~AtenDeriveIndexLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenDeriveIndexLayer>(new AtenDeriveIndexLayer(*this));
    }

    void setStep(int64_t step) { _step = step; }

    int64_t getStep() const { return _step; }

    void setStart(int64_t start) { _start = start; }

    int64_t getStart() const { return _start; }

    void printAttr() {
        Log::IR::I() << "    AtenDeriveIndexAttr      ";
        Log::IR::I() << "    start is                 "<< _start;
        Log::IR::I() << "    step is                  "<< _step;
    }

 private:
    int64_t _start = INT64_MIN;
    int64_t _step = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
