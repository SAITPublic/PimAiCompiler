#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenDropoutLayer : public NNLayer {
 public:
    AtenDropoutLayer() {}

    AtenDropoutLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenDropoutLayer(const AtenDropoutLayer& aten_drop_layer) :
        NNLayer(aten_drop_layer) {
        this->_proportion = aten_drop_layer._proportion;
        this->_train = aten_drop_layer._train;
    }

    virtual ~AtenDropoutLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenDropoutLayer>(new AtenDropoutLayer(*this));
    }

    void setProportion(double proportion) { _proportion = proportion; }
    void setTrain(int train) { _train = train; }
    double getProportion() { return _proportion; }
    int getTrain() { return _train; }

    void printAttr() {
        Log::IR::I() << "    AtenDropoutAttr            ";
        Log::IR::I() << "    proportion is              " << _proportion;
        Log::IR::I() << "    train value is             " << _train;
    }

 private:
    double _proportion = DBL_MAX;
    int  _train       = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
