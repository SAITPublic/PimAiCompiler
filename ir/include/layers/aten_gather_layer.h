#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenGatherLayer : public NNLayer
{
   public:
    AtenGatherLayer() {}

    AtenGatherLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenGatherLayer(const AtenGatherLayer &aten_gather_layer) : NNLayer(aten_gather_layer)
    {
        this->dim_ = aten_gather_layer.dim_;
        this->sparse_grad_ = aten_gather_layer.sparse_grad_;
    }

    virtual ~AtenGatherLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenGatherLayer>(new AtenGatherLayer(*this)); }

    void setDim(int dim) { dim_ = dim; }

    int getDim() const { return dim_; }

    void setSparseGrad(int sparse_grad) { sparse_grad_ = sparse_grad; }

    int getSparseGrad() const { return sparse_grad_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenGatherAttr      ";
        DLOG(INFO) << "    dim is              " << dim_;
        DLOG(INFO) << "    sparse_grad is      " << sparse_grad_;
    }

   private:
    int dim_ = INT32_MAX;
    int sparse_grad_ = INT32_MAX;
};

}  // namespace ir
}  // namespace nn_compiler
