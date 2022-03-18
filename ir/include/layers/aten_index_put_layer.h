#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenIndexPutLayer : public NNLayer {
 public:
    AtenIndexPutLayer() {}

    AtenIndexPutLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit AtenIndexPutLayer(const AtenIndexPutLayer &aten_index_put_layer) :
        NNLayer(aten_index_put_layer) {
        this->accumulate_ = aten_index_put_layer.accumulate_;
    }

    virtual ~AtenIndexPutLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenIndexPutLayer>(new AtenIndexPutLayer(*this));
    }

    void setAccumulate(int accumulate) { accumulate_ = accumulate; }

    int getAccumulate() const { return accumulate_; }

    void printAttr() {
        DLOG(INFO) << "    AtenIndexPutAttr             ";
        DLOG(INFO) << "    accumulate is                "<< accumulate_;
    }

 private:
    int accumulate_ = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
