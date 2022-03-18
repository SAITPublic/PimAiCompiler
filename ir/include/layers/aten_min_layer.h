#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenMinLayer : public NNLayer {
 public:
    AtenMinLayer() {}

    AtenMinLayer(std::string name, LayerType type)
        : NNLayer(name, type) {
    }

    explicit AtenMinLayer(const AtenMinLayer &aten_min_layer) :
        NNLayer(aten_min_layer) {
        this->_dim_or_y = aten_min_layer._dim_or_y;
        this->_keep_dim = aten_min_layer._keep_dim;
    }

    virtual ~AtenMinLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenMinLayer>(new AtenMinLayer(*this));
    }

    void setDimOrY(int dim_or_y) { _dim_or_y = dim_or_y; }

    void setKeepDim(int keep_dim) { _keep_dim = keep_dim; }

    int getDimOrY() { return _dim_or_y; }

    int getKeepDim() { return _keep_dim; }

    void printAttr() {
        DLOG(INFO) <<   " AtemMinAttr ";
        DLOG(INFO) <<   " dim_or_y is " << _dim_or_y;
        DLOG(INFO) <<   " keepdim is  " << _keep_dim;
    }

 private:
    int  _dim_or_y  = INT32_MAX;
    int  _keep_dim  = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
