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
        this->dim_ = aten_cat_layer.dim_;
        this->mem_layer_id_ = aten_cat_layer.mem_layer_id_;
    }

    virtual ~AtenCatLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenCatLayer>(new AtenCatLayer(*this)); }

    void setDim(int dim) { dim_ = dim; }
    int getDim() { return dim_; }

    void setMemLayerId(int64_t mem_layer_id) { mem_layer_id_ = mem_layer_id; }
    int64_t getMemLayerId() { return mem_layer_id_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenCatAttr       ";
        DLOG(INFO) << "    dim is            " << dim_;
    }

   private:
    int dim_ = INT32_MAX;
    int64_t mem_layer_id_ = -1;
};

}  // namespace ir
}  // namespace nn_compiler
