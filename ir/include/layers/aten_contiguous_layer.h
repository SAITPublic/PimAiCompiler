#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenContiguousLayer : public NNLayer {
 public:
    AtenContiguousLayer() {}

    AtenContiguousLayer(std::string name, LayerType type)
                     : NNLayer(name, type) {
    }

    explicit AtenContiguousLayer(const AtenContiguousLayer& aten_contiguous_layer) :
        NNLayer(aten_contiguous_layer) {
        this->memory_format_  = aten_contiguous_layer.memory_format_;
    }

    virtual ~AtenContiguousLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenContiguousLayer>(new AtenContiguousLayer(*this));
    }

    void setMemoryFormat(int memory_format) { memory_format_ = memory_format; }

    int getMemoryFormat() { return memory_format_; }

    void printAttr() {
        DLOG(INFO) << "    AtenContiguousAttr          ";
        DLOG(INFO) << "    memory_format is            " << memory_format_;
    }

 private:
    int memory_format_ = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
