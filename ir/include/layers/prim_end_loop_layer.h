#pragma once

#include "ir/include/layers/nn_layer.h"
#include "ir/include/types.h"
namespace nn_compiler
{
namespace ir
{
// def zeros(g, sizes, dtype, layout, device, pin_memory=False)

class PrimEndLoopLayer : public NNLayer
{
   public:
    /**
     * @brief PrimIfLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimEndLoopLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit PrimEndLoopLayer(const PrimEndLoopLayer& prim_end_loop_layer) : NNLayer(prim_end_loop_layer)
    {
        goto_layer_ = prim_end_loop_layer.goto_layer_;
    }

    virtual ~PrimEndLoopLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<PrimEndLoopLayer>(new PrimEndLoopLayer(*this)); }

    void setGotoLayer(int64_t goto_layer) { goto_layer_ = goto_layer; }

    int64_t getGotoLayer() const { return goto_layer_; }

    void printAttr()
    {
        DLOG(INFO) << "     PrimEndLoopAttr    ";
        DLOG(INFO) << "     goto_layer      " << goto_layer_;
    }

   private:
    int64_t goto_layer_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
