#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{

class PrimLoopLayer : public NNLayer
{
   public:
    /**
     * @brief PrimLoopLayer constructor
     * @param name the name of the layer
     * @param type the type of the layer
     */
    PrimLoopLayer(std::string name, LayerType type)
            : NNLayer(name, type) {
    }

    explicit PrimLoopLayer(const PrimLoopLayer& prim_loop_layer) : NNLayer(prim_loop_layer)
    {
        this->trip_count_ = prim_loop_layer.trip_count_;
        this->cond_ = prim_loop_layer.cond_;
        this->body_net_ = prim_loop_layer.body_net_;
        this->goto_layer_ = prim_loop_layer.goto_layer_;
    }

    virtual ~PrimLoopLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<PrimLoopLayer>(new PrimLoopLayer(*this)); }

    void setTripCount(const int64_t trip_count) { trip_count_ = trip_count; }

    void setCond(int64_t cond) { cond_ = cond; }

    void setBodyNet(const std::string body_net) { body_net_ = body_net; }

    const int64_t getTripCount() { return trip_count_; }

    int64_t getCond() { return cond_; }

    const std::string getBodyNet() { return body_net_; }

    void setGotoLayer(int64_t goto_layer) { goto_layer_ = goto_layer; }

    int64_t getGotoLayer() { return goto_layer_; }

    void printAttr()
    {
        Log::IR::I() << "     PrimLoopAttr    ";
        Log::IR::I() << "     body_net        " << body_net_;
        Log::IR::I() << "     trip_count      " << trip_count_;
        Log::IR::I() << "     cond            " << cond_;
    }

   private:
    int64_t trip_count_ = INT64_MIN;
    int64_t cond_ = INT64_MIN;
    std::string body_net_;

    int64_t goto_layer_;
};

}  // namespace ir
}  // namespace nn_compiler
