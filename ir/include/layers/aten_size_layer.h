#pragma once

#include "ir/include/layers/nn_layer.h"

namespace nn_compiler
{
namespace ir
{
// def size(g, self, dim=None)
class AtenSizeLayer : public NNLayer
{
   public:
    AtenSizeLayer() {}

    AtenSizeLayer(std::string name, LayerType type) : NNLayer(name, type) {}

    explicit AtenSizeLayer(const AtenSizeLayer& aten_size_layer) : NNLayer(aten_size_layer)
    {
        this->dim_ = aten_size_layer.getDim();
    }

    virtual ~AtenSizeLayer() {}

    virtual std::shared_ptr<NNLayer> clone() { return std::shared_ptr<AtenSizeLayer>(new AtenSizeLayer(*this)); }

    void setDim(int64_t dim) { dim_ = dim; }

    int64_t getDim() const { return dim_; }

    void printAttr()
    {
        DLOG(INFO) << "    AtenSizeAttr      ";
        DLOG(INFO) << "    dim is           " << dim_;
    }

   private:
    int64_t dim_ = INT64_MIN;
};

}  // namespace ir
}  // namespace nn_compiler
