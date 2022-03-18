#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class PrimLoopIndexLayer : public NNLayer
{
   public:
    /**
     * @brief PrimLoopIndexLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimLoopIndexLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimLoopIndexLayer(const PrimLoopIndexLayer& prim_loop_index_layer) : NNLayer(prim_loop_index_layer)
    {
        index_ = prim_loop_index_layer.index_;
    }

    virtual ~PrimLoopIndexLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<PrimLoopIndexLayer>(new PrimLoopIndexLayer(*this));
    }

    void setIndex(int64_t index) { index_ = index; }

    int64_t getIndex() { return index_; }

    void printAttr() { DLOG(INFO) << "     PrimLoopIndexAttr    "; }

   private:
    int64_t index_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
