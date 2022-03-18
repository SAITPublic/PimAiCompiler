#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
class AtenLogSoftmaxLayer : public NNLayer
{
   public:
    AtenLogSoftmaxLayer() {}

    AtenLogSoftmaxLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenLogSoftmaxLayer(const AtenLogSoftmaxLayer &aten_log_softmax_layer) : NNLayer(aten_log_softmax_layer)
    {
        this->dim_ = aten_log_softmax_layer.dim_;
        this->dtype_ = aten_log_softmax_layer.dtype_;
    }

    virtual ~AtenLogSoftmaxLayer() {}

    virtual std::shared_ptr<NNLayer> clone()
    {
        return std::shared_ptr<AtenLogSoftmaxLayer>(new AtenLogSoftmaxLayer(*this));
    }

    void setDim(int64_t dim) { dim_ = dim; }

    void setDType(int64_t dtype) { dtype_ = dtype; }

    int64_t getDim() const { return dim_; }

    int64_t getDType() const { return dtype_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenLogSoftmaxAttr      ";
        DLOG(INFO) << "    dim is                  " << dim_;
        DLOG(INFO) << "    dtype is                " << dtype_;
    }

   private:
    int64_t dim_ = INT64_MIN;
    int64_t dtype_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
