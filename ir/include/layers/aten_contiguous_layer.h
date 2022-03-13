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
        this->_memory_format  = aten_contiguous_layer._memory_format;
    }

    virtual ~AtenContiguousLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenContiguousLayer>(new AtenContiguousLayer(*this));
    }

    void setMemoryFormat(int memory_format) { _memory_format = memory_format; }

    int getMemoryFormat() { return _memory_format; }

    void printAttr() {
        Log::IR::I() << "    AtenContiguousAttr          ";
        Log::IR::I() << "    memory_format is            " << _memory_format;
    }

 private:
    int _memory_format = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
