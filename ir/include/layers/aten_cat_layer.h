#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenCatLayer : public NNLayer
{
   public:
    AtenCatLayer() {}

    AtenCatLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenCatLayer(const AtenCatLayer& aten_cat_layer) : NNLayer(aten_cat_layer)
    {
        this->_dim = aten_cat_layer._dim;
    }

    virtual ~AtenCatLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenCatLayer>(new AtenCatLayer(*this)); }

    void setDim(int dim) { _dim = dim; }
    int getDim() { return _dim; }

    void setMemLayerId(int64_t mem_layer_id) { mem_layer_id_ = mem_layer_id; }
    int64_t getMemLayerId() { return mem_layer_id_; }

    void printAttr()
    {
        Log::IR::I() << "    AtenCatAttr       ";
        Log::IR::I() << "    dim is            " << _dim;
    }

   private:
    int _dim = INT32_MAX;
    int64_t mem_layer_id_ = -1;
};

}  // namespace ir
}  // namespace nn_compiler
