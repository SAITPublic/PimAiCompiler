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
        this->proportion_ = aten_drop_layer.proportion_;
        this->train_ = aten_drop_layer.train_;
    }

    virtual ~AtenDropoutLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenDropoutLayer>(new AtenDropoutLayer(*this));
    }

    void setProportion(double proportion) { proportion_ = proportion; }
    void setTrain(int train) { train_ = train; }
    double getProportion() { return proportion_; }
    int getTrain() { return train_; }

    void printAttr() {
        DLOG(INFO) << "    AtenDropoutAttr            ";
        DLOG(INFO) << "    proportion is              " << proportion_;
        DLOG(INFO) << "    train value is             " << train_;
    }

 private:
    double proportion_ = DBL_MAX;
    int  train_       = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
