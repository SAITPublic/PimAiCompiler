#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {


class AtenChunkLayer : public NNLayer {
 public:
    AtenChunkLayer() {}

    AtenChunkLayer(std::string name, LayerType type)
                     : NNLayer(name, type) {
    }

    explicit AtenChunkLayer(const AtenChunkLayer& aten_chunk_layer) : NNLayer(aten_chunk_layer) {
        this->_chunks  = aten_chunk_layer._chunks;
        this->_dim = aten_chunk_layer._dim;
    }

    virtual ~AtenChunkLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return  std::shared_ptr<AtenChunkLayer>(new AtenChunkLayer(*this));
    }

    void setChunks(int chunks) { _chunks = chunks; }

    int getChunks() { return _chunks; }

    void setDim(int dim) { _dim = dim; }

    int getDim() { return _dim; }

    void printAttr() {
        Log::IR::I() << "    AtenChunkAttr          ";
        Log::IR::I() << "    chunks is              " << _chunks;
        Log::IR::I() << "    dim is                 " << _dim;
    }

 private:
    int _chunks = INT32_MAX;
    int _dim    = INT32_MAX;
};

} // namespace ir
} // namespace nn_compiler
