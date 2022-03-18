
#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenIndexSelectLayer : public NNLayer {
 public:
    AtenIndexSelectLayer() {}

    AtenIndexSelectLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenIndexSelectLayer(const AtenIndexSelectLayer& aten_index_select_layer) :
        NNLayer(aten_index_select_layer) {
        this->dim_ = aten_index_select_layer.dim_;
    }

    virtual ~AtenIndexSelectLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenIndexSelectLayer>(new AtenIndexSelectLayer(*this));
    }

    void setDim(int dim) { dim_ = dim; }

    int getDim() const { return dim_; }

    void printAttr() {
        DLOG(INFO) << "    AtenIndexSelectAttr     ";
        DLOG(INFO) << "    dim is                  "<< dim_;
    }

 private:
    int dim_ = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
