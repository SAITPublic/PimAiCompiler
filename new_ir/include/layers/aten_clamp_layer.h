#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenClampLayer : public NNLayer {
 public:
    AtenClampLayer() {}

    AtenClampLayer(std::string name, std::string type)
                     : NNLayer(name, type) {
    }

    explicit AtenClampLayer(const AtenClampLayer& aten_clamp_layer) : NNLayer(aten_clamp_layer) {
        this->_min  = aten_clamp_layer._min;
        this->_max = aten_clamp_layer._max;
    }

    virtual ~AtenClampLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenClampLayer>(new AtenClampLayer(*this));
    }

    void setMin(int min) { _min = min; }

    int getMin() { return _min; }

    void setMax(int max) { _max = max; }

    int getMax() { return _max; }

    void printAttr() {
        Log::IR::I() << "    AtenClampAttr          ";
        Log::IR::I() << "    min is                 " << _min;
        Log::IR::I() << "    max is                 " << _max;
    }

 private:
    int _min = INT32_MAX;
    int _max = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
