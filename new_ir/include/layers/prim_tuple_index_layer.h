#pragma once

#include "new_ir/include/layers/nn_layer.h"

namespace nn_compiler {
namespace ir {

class PrimTupleIndexLayer : public NNLayer {
 public:
    /**
     * @brief PrimTupleIndexLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimTupleIndexLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit PrimTupleIndexLayer(const PrimTupleIndexLayer& prim_tuple_index_layer) :
        NNLayer(prim_tuple_index_layer) {
        this->_index = prim_tuple_index_layer._index;
    }

    virtual ~PrimTupleIndexLayer() {}

    virtual std::shared_ptr<NNLayer> clone() {
        return std::shared_ptr<PrimTupleIndexLayer>(new PrimTupleIndexLayer(*this));
    }

    int64_t getIndex() const { return _index; }

    void setIndex(int64_t index) { _index = index; }

    void printAttr() {
        Log::IR::I() << "   PrimTupleIndexAttr    ";
        Log::IR::I() << "   index is    " << _index;
    }

 private:
    int64_t _index = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
