#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class PrimTupleIndexLayer : public NNLayer
{
   public:
    /**
     * @brief PrimTupleIndexLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimTupleIndexLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimTupleIndexLayer(const PrimTupleIndexLayer& prim_tuple_index_layer) : NNLayer(prim_tuple_index_layer)
    {
        this->index_ = prim_tuple_index_layer.index_;
    }

    virtual ~PrimTupleIndexLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<PrimTupleIndexLayer>(new PrimTupleIndexLayer(*this));
    }

    int64_t getIndex() const { return index_; }

    void setIndex(int64_t index) { index_ = index; }

    void printAttr()
    {
        DLOG(INFO) << "   PrimTupleIndexAttr    ";
        DLOG(INFO) << "   index is    " << index_;
    }

   private:
    int64_t index_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
