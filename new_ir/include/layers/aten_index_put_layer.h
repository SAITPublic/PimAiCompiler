#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class AtenIndexPutLayer : public NNLayer {
 public:
    AtenIndexPutLayer() {}

    AtenIndexPutLayer(std::string name, std::string type)
            : NNLayer(name, type) {
    }

    explicit AtenIndexPutLayer(const AtenIndexPutLayer &aten_index_put_layer) :
        NNLayer(aten_index_put_layer) {
        this->_accumulate = aten_index_put_layer._accumulate;
    }

    virtual ~AtenIndexPutLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<AtenIndexPutLayer>(new AtenIndexPutLayer(*this));
    }

    void setAccumulate(int accumulate) { _accumulate = accumulate; }

    int getAccumulate() const { return _accumulate; }

    void printAttr() {
        Log::IR::I() << "    AtenIndexPutAttr             ";
        Log::IR::I() << "    accumulate is                "<< _accumulate;
    }

 private:
    int _accumulate = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
