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
        this->_trip_count = prim_loop_layer._trip_count;
        this->_cond = prim_loop_layer._cond;
        this->_body_net = prim_loop_layer._body_net;
    }

    virtual ~PrimLoopLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<PrimLoopLayer>(new PrimLoopLayer(*this)); }

    void setTripCount(const int64_t trip_count) { _trip_count = trip_count; }

    void setCond(int64_t cond) { _cond = cond; }

    void setBodyNet(const std::string body_net) { _body_net = body_net; }

    const int64_t getTripCount() { return _trip_count; }

    int64_t getCond() { return _cond; }

    const std::string getBodyNet() { return _body_net; }

    void setGotoLayer(int64_t goto_layer) { goto_layer_ = goto_layer; }

    int64_t getGotoLayer() { return goto_layer_; }

    void printAttr()
    {
        Log::IR::I() << "     PrimLoopAttr    ";
        Log::IR::I() << "     body_net        " << _body_net;
        Log::IR::I() << "     trip_count      " << _trip_count;
        Log::IR::I() << "     cond            " << _cond;
    }

   private:
    int64_t _trip_count = INT64_MIN;
    int64_t _cond = INT64_MIN;
    std::string _body_net;

    int64_t goto_layer_;
};

}  // namespace ir
}  // namespace nn_compiler
